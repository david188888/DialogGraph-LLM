import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from tqdm import tqdm
import logging
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys

from utils.graph import DialogueGraphModel
from utils.dataloader import AudioSegmentDataset, LabelBalancedSampler, DataLoaderConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveThresholdTrainer:
    """适应性音频分段训练器，支持Qwen Omni模型训练"""
    
    def __init__(
        self, 
        model,
        processor,
        graph_config, 
        output_dir, 
        device="cuda",
        learning_rate=5e-5,
        weight_decay=0.01,
        segment_loss_weight=0.5,
        num_classes=4
    ):
        """
        初始化训练器
        
        参数:
            model: Qwen Omni 模型
            processor: Qwen Omni 处理器
            graph_config: 图模型配置
            output_dir: 输出目录
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            segment_loss_weight: 片段损失权重
            num_classes: 分类任务的类别数量
        """
        self.model = model
        self.processor = processor
        self.output_dir = output_dir
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.segment_loss_weight = segment_loss_weight
        self.num_classes = num_classes
        
        # 初始化图模型
        self.graph_model = DialogueGraphModel(**graph_config).to(device)
        
        # 初始化优化器
        self._init_optimizer()
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 收集需要优化的参数
        optimizer_grouped_parameters = [
            # 模型参数
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad],
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate
            },
            # 图模型参数
            {
                "params": self.graph_model.parameters(),
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate
            }
        ]
        
        # 初始化AdamW优化器
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )
    
    def train(
        self, 
        train_data_path, 
        val_data_path=None, 
        num_epochs=5, 
        gradient_accumulation_steps=4,
        log_interval=10,
        eval_interval=100,
        save_interval=1000,
        use_wandb=False,
        augment_segments=True,
        system_prompt=None,
        num_segments=None,
        batch_size=8,
        use_balanced_sampling=True
    ):
        """
        训练模型，使用新的音频分段数据加载器
        
        参数:
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径
            num_epochs: 训练轮数
            gradient_accumulation_steps: 梯度累积步数
            log_interval: 日志记录间隔
            eval_interval: 评估间隔
            save_interval: 保存间隔
            use_wandb: 是否使用wandb记录实验
            augment_segments: 是否增强分段
            system_prompt: 系统提示文本
            num_segments: 每个音频切分的片段数量，None表示随机
            batch_size: 批次大小
            use_balanced_sampling: 是否使用平衡采样
        """
        # 初始化wandb
        if use_wandb:
            wandb.init(
                project="audio-llm-telemarketing",
                config={
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "segment_loss_weight": self.segment_loss_weight,
                    "num_epochs": num_epochs,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "augment_segments": augment_segments,
                    "num_segments": num_segments or "random 3-5"
                }
            )
        
        # 1. 创建配置实例
        train_config = DataLoaderConfig(
            data_path=train_data_path,
            labels_file='migrated_labels.csv',
            cache_dir='features_cache',
            shuffle=not use_balanced_sampling, # 如果使用平衡采样，shuffle为False
            num_workers=0, # 避免多进程问题
            system_prompt=system_prompt,
            balance_labels=use_balanced_sampling,
            model_path=self.processor.name_or_path if hasattr(self.processor, 'name_or_path') else None
 
        )

        # 2. 创建 Dataset 实例
        train_dataset = AudioSegmentDataset(
            data_path=train_config.data_path,
            model_path=train_config.model_path,
            labels_file=train_config.labels_file,
            cache_dir=train_config.cache_dir
        )

        # 3. 创建 DataLoader
        train_sampler = None
        if train_config.balance_labels:
            train_sampler = LabelBalancedSampler(
                dataset=train_dataset,
                batch_size=train_config.batch_size,
                front_dense_ratio=train_config.front_dense_ratio,
                dense_factor=train_config.dense_factor
            )
            train_config.shuffle = False # 使用 sampler 时 shuffle 必须为 False
            print(f"已启用标签均衡采样，前{int(len(train_dataset) * train_config.front_dense_ratio)}个样本中有标签样本密度为标准的{train_config.dense_factor}倍")

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            sampler=train_sampler,
            shuffle=train_config.shuffle,
            collate_fn=train_dataset.collate_fn,
            num_workers=train_config.num_workers
        )
        
        val_loader = None
        if val_data_path:
            # 创建验证集配置 (通常不需要平衡采样)
            val_config = DataLoaderConfig(
                data_path=val_data_path,
                labels_file='migrated_labels.csv',
                cache_dir='features_cache',
                batch_size=batch_size, # 使用相同的 batch_size
                shuffle=False, # 验证集不打乱
                num_workers=0,
                system_prompt=system_prompt,
                balance_labels=True, # 验证集不使用平衡采样
                model_path=train_config.model_path
            )
            # 创建验证集 Dataset
            val_dataset = AudioSegmentDataset(
                data_path=val_config.data_path,
                model_path=val_config.model_path,
                labels_file=val_config.labels_file,
                cache_dir=val_config.cache_dir
            )
            # 创建验证集 DataLoader
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_config.batch_size,
                shuffle=val_config.shuffle,
                collate_fn=val_dataset.collate_fn,
                num_workers=val_config.num_workers
            )
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 训练循环
        global_step = 0
        total_steps = len(train_loader) * num_epochs
        
        for epoch in range(num_epochs):
            self.model.train()
            self.graph_model.train()
            
            epoch_loss = 0
            original_loss = 0
            segment_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # 每个批次只有一个样本(batch_size=1)
                num_batches += 1
                
                # 提取批次数据并移动到设备
                # 适配新的dataloader返回格式
                segment_features = batch['segment_features'].to(self.device)  # [batch, max_num_segments, max_segment_len, feat_dim]
                segment_attention_mask = batch['segment_attention_mask'].to(self.device)  # [batch, max_num_segments, max_segment_len]
                labels = batch['label']  
                num_segments = batch['num_segments']
                speakers = batch['speaker']  # 说话者ID列表
                
                # 将字符串标签转换为数值标签（如果需要）
                numeric_labels = []
                for label in labels:
                    if label is None or label == '':
                        numeric_labels.append(-1)  # 无标签
                    else:
                        try:
                            # 尝试将标签转换为数字
                            numeric_labels.append(int(label))
                        except (ValueError, TypeError):
                            # 如果无法转换为数字，需要维护一个标签映射字典
                            # 这里简化处理，将所有非数字标签视为0
                            numeric_labels.append(0)
                
                numeric_labels_tensor = torch.tensor(numeric_labels, dtype=torch.long, device=self.device)
                
                # 使用图网络处理segment_features
                try:
                    # 使用图模型处理segment_features (得到图的嵌入表示)
                    graph_embedding = self.graph_model(
                        segment_features,  # [batch, max_num_segments, max_segment_len, feat_dim]
                        speakers,          # [batch, max_num_segments] 或 List[List]
                        attention_masks=segment_attention_mask  # [batch, max_num_segments, max_segment_len]
                    )
                    
                    # 图嵌入形状: [batch_size, output_dim]
                    graph_feature = graph_embedding
                    
                except Exception as e:
                    logger.error(f"图网络处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    graph_feature = None
                
                # 构建适合模型的输入格式
                # 优先使用图特征，如果没有则使用原始特征
                try:
                    # 构造模型输入
                    inputs = {
                        "input_ids": torch.ones(1, 4).long().to(self.device),  # 占位符
                        "attention_mask": torch.ones(1, 4).to(self.device),    # 占位符
                        "labels": numeric_labels_tensor                        # 数值化的标签
                    }
                    
                    # 添加图特征
                    if graph_feature is not None:
                        inputs["features"] = graph_feature
                    
                    original_outputs = self.model(**inputs)
                    orig_loss = original_outputs.loss
                    original_loss += orig_loss.item()
                except Exception as e:
                    logger.error(f"模型处理图特征时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    orig_loss = torch.tensor(0.0, device=self.device)
                
                # 2. 处理音频片段特征
                segments_batch_loss = 0
                num_valid_segments = 0
                
                # 针对每个批次样本处理片段特征
                for batch_idx in range(segment_features.size(0)):
                    # 获取实际有效的片段数
                    actual_num_segs = num_segments[batch_idx]
                    
                    # 使用 get_node_embeddings 获取节点特征
                    try:
                        # 提取单个样本的数据
                        sample_features = segment_features[batch_idx:batch_idx+1]  # [1, max_num_segments, max_segment_len, feat_dim]
                        sample_mask = segment_attention_mask[batch_idx:batch_idx+1]  # [1, max_num_segments, max_segment_len]
                        sample_speakers = [speakers[batch_idx]]  # 单个样本的说话者列表
                        
                        # 获取节点特征
                        node_features = self.graph_model.get_node_embeddings(
                            sample_features,
                            sample_speakers,
                            attention_masks=sample_mask
                        )
                        # node_features: [max_num_segments, max_segment_len, output_dim]
                        
                        # 只处理有效的片段
                        valid_node_features = node_features[:actual_num_segs]
                        
                        # 对每个有效片段进行处理
                        for seg_idx in range(actual_num_segs):
                            # 获取当前片段特征并进行平均池化
                            seg_feature = valid_node_features[seg_idx]  # [max_segment_len, output_dim]
                            
                            # 使用有效的掩码计算平均值
                            seg_mask = sample_mask[0, seg_idx].bool()  # [max_segment_len]
                            if seg_mask.sum() > 0:  # 有有效token
                                # 使用掩码获取有效的特征并平均
                                valid_indices = seg_mask.nonzero().squeeze(-1)
                                valid_features = seg_feature[valid_indices]
                                segment_embedding = valid_features.mean(dim=0)  # [output_dim]
                                
                                # 构造片段输入
                                segment_inputs = {
                                    "input_ids": torch.ones(1, 4).long().to(self.device),  # 占位符
                                    "attention_mask": torch.ones(1, 4).to(self.device),    # 占位符
                                    "labels": numeric_labels_tensor[batch_idx:batch_idx+1],  # 当前样本的标签
                                    "features": segment_embedding.unsqueeze(0)  # [1, output_dim]
                                }
                                
                                # 前向传播
                                segment_outputs = self.model(**segment_inputs)
                                segments_batch_loss += segment_outputs.loss
                                num_valid_segments += 1
                    except Exception as e:
                        logger.error(f"处理单个样本的节点特征时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # 计算平均片段损失
                if num_valid_segments > 0:
                    avg_segment_loss = segments_batch_loss / num_valid_segments
                    segment_loss += avg_segment_loss.item()
                else:
                    avg_segment_loss = torch.tensor(0.0, device=self.device)
                
                # 3. 计算总损失 - 加权组合
                loss = orig_loss + self.segment_loss_weight * avg_segment_loss
                
                # 累积梯度
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                # 梯度累积
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        "loss": epoch_loss / (step + 1),
                        "orig_loss": original_loss / num_batches,
                        "segment_loss": segment_loss / num_batches
                    })
                    
                    # 记录日志
                    if global_step % log_interval == 0 and use_wandb:
                        wandb.log({
                            "loss": epoch_loss / (step + 1),
                            "original_loss": original_loss / num_batches,
                            "segment_loss": segment_loss / num_batches,
                            "global_step": global_step,
                            "progress": global_step / total_steps
                        })
                    
                    # 评估模型
                    if val_data_path and global_step % eval_interval == 0:
                        eval_results = self.evaluate(val_loader)
                        
                        logger.info(f"Evaluation at step {global_step}:")
                        for metric, value in eval_results.items():
                            logger.info(f"  {metric}: {value:.4f}")
                        
                        if use_wandb:
                            wandb.log(eval_results)
                        
                        # 恢复训练模式
                        self.model.train()
                        self.graph_model.train()
                    
                    # 保存模型
                    if global_step % save_interval == 0:
                        self.save_model(os.path.join(self.output_dir, f"checkpoint-{global_step}"))
            
            # 每个epoch结束后记录日志
            epoch_avg_loss = epoch_loss / len(train_loader)
            original_avg_loss = original_loss / num_batches
            segment_avg_loss = segment_loss / num_batches
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed:")
            logger.info(f"  Average loss: {epoch_avg_loss:.4f}")
            logger.info(f"  Original loss: {original_avg_loss:.4f}")
            logger.info(f"  Segment loss: {segment_avg_loss:.4f}")
            
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_loss": epoch_avg_loss,
                    "epoch_original_loss": original_avg_loss,
                    "epoch_segment_loss": segment_avg_loss
                })
            
            # 保存每个epoch的模型
            self.save_model(os.path.join(self.output_dir, f"epoch-{epoch+1}"))
        
        # 保存最终模型
        self.save_model(os.path.join(self.output_dir, "final-model"))
        
        if use_wandb:
            wandb.finish()
    
    def evaluate(self, val_loader):
        """
        评估模型
        
        参数:
            val_loader: 验证数据加载器
            
        返回:
            评估结果字典
        """
        self.model.eval()
        self.graph_model.eval()
        
        val_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                num_batches += 1
                
                # 提取批次数据并移动到设备 - 适配新的dataloader格式
                phone_ids = batch['phone_ids']
                segment_features = batch['segment_features'].to(self.device)
                segment_attention_mask = batch['segment_attention_mask'].to(self.device)
                labels = batch['label']
                speakers = batch['speaker']
                
                # 将字符串标签转换为数值标签
                numeric_labels = []
                for label in labels:
                    if label is None or label == '':
                        numeric_labels.append(-1)  # 无标签
                    else:
                        try:
                            numeric_labels.append(int(label))
                        except (ValueError, TypeError):
                            numeric_labels.append(0)  # 默认0
                
                numeric_labels_tensor = torch.tensor(numeric_labels, dtype=torch.long, device=self.device)
                
                # 使用图网络处理
                try:
                    # 获取图嵌入
                    graph_embedding = self.graph_model(
                        segment_features,
                        speakers,
                        attention_masks=segment_attention_mask
                    )
                    
                    # 构造模型输入
                    inputs = {
                        "input_ids": torch.ones(len(numeric_labels), 4).long().to(self.device),  # 占位符
                        "attention_mask": torch.ones(len(numeric_labels), 4).to(self.device),    # 占位符
                        "labels": numeric_labels_tensor,  # 当前批次的标签
                        "features": graph_embedding  # [batch, output_dim]
                    }
                    
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    # 如果模型输出包含logits，则获取预测结果
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(numeric_labels_tensor.cpu().numpy())
                except Exception as e:
                    logger.error(f"评估时处理批次出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 计算平均损失
        avg_loss = val_loss / max(num_batches, 1)
        
        # 计算评估指标
        results = {
            "eval_loss": avg_loss,
        }
        
        # 如果有预测结果，计算准确率
        if all_preds and all_labels and len(all_preds) == len(all_labels):
            # 过滤掉无效标签(-1)
            valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
            if valid_indices:
                valid_preds = [all_preds[i] for i in valid_indices]
                valid_labels = [all_labels[i] for i in valid_indices]
                
                accuracy = accuracy_score(valid_labels, valid_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    valid_labels, valid_preds, average='weighted'
                )
                
                results.update({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })
        
        return results
    
    def save_model(self, output_path):
        """
        保存模型
        
        参数:
            output_path: 输出路径
        """
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(output_path)
        
        # 保存处理器
        self.processor.save_pretrained(output_path)
        
        # 保存图模型
        torch.save(self.graph_model.state_dict(), os.path.join(output_path, "graph_model.pt"))
        
        # 保存训练配置
        with open(os.path.join(output_path, "trainer_config.json"), "w") as f:
            import json
            json.dump({
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "segment_loss_weight": self.segment_loss_weight
            }, f)
        
        logger.info(f"模型保存到: {output_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        参数:
            model_path: 模型路径
        """
        # 加载图模型
        self.graph_model.load_state_dict(torch.load(os.path.join(model_path, "graph_model.pt")))
        
        logger.info(f"模型从 {model_path} 加载成功")


def main():
    """主函数"""
    # 添加路径
    sys.path.append('/data/shared/Qwen/ECAI')
    from ECAI.qwen2_5_omni_light import Qwen25OmniLightProcessor, Qwen2_5OmniTextOnlyModel
    import torch
    
    # 配置参数
    model_path = "/data/shared/Qwen/models/Qwen2.5-Omni-7B"
    
    # 加载processor
    processor = Qwen25OmniLightProcessor.from_pretrained(model_path)
    
    # 加载模型
    model = Qwen2_5OmniTextOnlyModel.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    # 更新图模型配置 - 与graph.py匹配的配置
    graph_config = {
        "token_embedding_dim": 768,  # 输入token嵌入维度
        "output_dim": 128,          # 输出维度
        "num_heads": 4,             # 注意力头数量
        "speaker_embedding_dim": 64, # 说话者嵌入维度
        "num_speakers": None,       # 动态说话者嵌入
        "num_layers": 2,            # GAT层数
        "dropout": 0.2,             # Dropout概率
        "similarity_threshold": 0.5,# 相似度阈值
        "context_window_size": 4,   # 上下文窗口大小
    }
    
    output_dir = "./outputs"
    
    # 数据路径
    train_data_path = "./data/train"
    val_data_path = "./data/val"
    preprocessed_path = "./data/segments"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preprocessed_path, exist_ok=True)
    
    # 初始化训练器
    trainer = AdaptiveThresholdTrainer(
        model=model,
        processor=processor,
        graph_config=graph_config,
        output_dir=output_dir,
        learning_rate=2e-5,
        segment_loss_weight=0.5,  # 片段损失权重
        num_classes=4
    )
    
    # 系统提示文本
    system_prompt = "You are a helpful assistant specialized in analyzing customer conversations."
    
    # 训练模型
    trainer.train(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        num_epochs=3,
        gradient_accumulation_steps=4,
        preprocessed_path=preprocessed_path,
        augment_segments=True,
        system_prompt=system_prompt,
        use_wandb=True
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--augment-segments', action='store_true', default=True,
                        help='是否对音频分段进行增强')
    parser.add_argument('--num-segments', type=int, default=None,
                        help='每个音频切分的片段数量，None表示随机')
    parser.add_argument('--system-prompt', type=str, 
                        default="You are a helpful assistant specialized in analyzing customer conversations.",
                        help='系统提示文本')
    
    args = parser.parse_args()
    
    main()
