import sys
import os

# 添加绝对路径
sys.path.append('/data/shared/Qwen')
# 添加相对路径（向上一级）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


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
# from transformers import (
#     Qwen2_5OmniThinkerConfig
# )
from transformers import AutoConfig
import numpy as np # 添加 numpy
from itertools import cycle # 用于处理不同长度的数据加载器


from ECAI.qwen2_5_omni_light import Qwen2_5OmniLightProcessor, Qwen2_5OmniLightForConditionalGeneration, Qwen2_5OmniTextOnlyModel


from utils.graph import DialogueGraphModel
from utils.dataloader import AudioSegmentDataset, LabelBalancedSampler, DataLoaderConfig





# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveThresholdTrainer:
    """适应性音频分段训练器，支持Qwen Omni模型训练，并集成半监督自适应阈值策略"""

    def __init__(
        self,
        model,
        processor,
        graph_config,
        output_dir,
        device="cuda:0",
        learning_rate=5e-5,
        weight_decay=0.01,
        # segment_loss_weight=0.5, # 暂时未使用，先注释掉
        num_classes=4,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        # --- 半监督相关参数 ---
        use_semi_supervised=False,
        lambda_ema=0.9,          # EMA动量参数
        initial_tau=0.8,         # 初始全局阈值
        top_k_percent=0.1,       # 选择伪标签样本的百分比 (用于更新数据集)
        update_dataset_freq=1    # 更新有标签数据集的频率（epoch）
    ):
        """
        初始化训练器

        参数:
            model: Qwen Omni backbone
            processor: Qwen Omni 处理器
            graph_config: 图模型配置
            output_dir: 输出目录
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            # segment_loss_weight: 片段损失权重
            num_classes: 分类任务的类别数量
            use_lora: 是否使用LoRA微调
            lora_r: LoRA的秩
            lora_alpha: LoRA的放大系数
            lora_dropout: LoRA的dropout概率
            use_semi_supervised: 是否启用半监督学习
            lambda_ema: EMA动量参数
            initial_tau: 初始全局阈值
            top_k_percent: 选择伪标签样本的百分比
            update_dataset_freq: 更新有标签数据集的频率（epoch）
        """
        self.model = model
        self.processor = processor
        self.output_dir = output_dir
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # self.segment_loss_weight = segment_loss_weight # 暂时未使用
        self.num_classes = num_classes
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # --- 半监督相关参数 ---
        self.use_semi_supervised = use_semi_supervised
        self.lambda_ema = lambda_ema
        self.initial_tau = initial_tau
        self.top_k_percent = top_k_percent
        self.update_dataset_freq = update_dataset_freq
        # 初始化自适应阈值参数 (仅在半监督模式下)
        if self.use_semi_supervised:
            self.tau = torch.tensor(self.initial_tau, device=self.device)
            self.p_tilde = torch.ones(self.num_classes, device=self.device) / self.num_classes
            self.tau_c = self.tau * (self.p_tilde / torch.max(self.p_tilde)) # 初始类别阈值

        # 如果使用LoRA，则应用LoRA配置
        if self.use_lora:
            logger.info("使用LoRA微调...")
            # 准备模型进行LoRA微调
            # 定义LoRA配置
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            lora_config = LoraConfig(
                r=lora_r,                            # LoRA的秩
                lora_alpha=lora_alpha,               # LoRA的放大系数
                target_modules=target_modules,       # 要应用LoRA的模块
                lora_dropout=lora_dropout,           # LoRA的dropout概率
                bias="none",                         # 不更新bias参数
                task_type="CAUSAL_LM",               # 任务类型
                inference_mode=False,                # 非推理模式
            )

            try:
                # 准备模型进行训练 (移除 prepare_model_for_kbit_training)
                # self.model = prepare_model_for_kbit_training(self.model)
                # 转换为PEFT模型
                self.model = get_peft_model(self.model, lora_config)
                # 打印可训练参数数量
                # self.model.print_trainable_parameters()
                logger.info("LoRA配置应用成功！")
            except Exception as e:
                logger.error(f"应用LoRA配置时出错: {e}")
                import traceback
                traceback.print_exc()
                logger.warning("将使用全参数微调...")
                self.use_lora = False

        # 初始化图模型
        self.graph_model = DialogueGraphModel(**graph_config).to(device)
        
        # 检查图模型是否需要转换为bfloat16精度
        if hasattr(self.model, 'dtype') and self.model.dtype == torch.bfloat16:
            logger.info("将图模型转换为bfloat16精度以匹配主模型")
            # 直接转换整个模型
            self.graph_model = self.graph_model.to(torch.bfloat16)
            # 手动设置 dtype 属性以供检查
            self.graph_model.dtype = torch.bfloat16 

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

    # --- 半监督辅助函数 ---
    def _update_adaptive_thresholds(self, Q_u):
        """
        更新自适应阈值 (内部方法)

        参数:
            Q_u: 无标签数据的预测概率 [batch_size, num_classes]
        """

        # 1. 计算每个样本的最大预测概率
        max_probs, _ = torch.max(Q_u, dim=1)  

        # 2. 更新全局置信度阈值（EMA）
        batch_avg_confidence = torch.mean(max_probs)
        self.tau = self.lambda_ema * self.tau + (1 - self.lambda_ema) * batch_avg_confidence

        # 3. 更新类别平均预测概率（EMA）
        batch_class_avg = torch.mean(Q_u, dim=0)  # [num_classes]
        self.p_tilde = self.lambda_ema * self.p_tilde + (1 - self.lambda_ema) * batch_class_avg

        # 4. 计算每个类别的自适应阈值
        max_class_prob = torch.max(self.p_tilde)
        # 防止除以零
        self.tau_c = (self.p_tilde / (max_class_prob + 1e-8)) * self.tau

        # 限制阈值在(0, 1)之间
        self.tau_c = torch.clamp(self.tau_c, min=1e-8, max=1.0 - 1e-8)
        self.tau = torch.clamp(self.tau, min=1e-8, max=1.0 - 1e-8)

        return self.tau, self.p_tilde, self.tau_c

    def _generate_pseudo_labels(self, Q_u):
        """
        生成伪标签 (内部方法)，对于有多个超过阈值的类别，只保留概率最高的一个

        参数:
            Q_u: 无标签数据的预测概率 [batch_size, num_classes]

        返回:
            P_u: 生成的伪标签 [batch_size, num_classes] (0.0或1.0)
            mask: 有效伪标签的掩码 [batch_size] (0.0或1.0)
        """
        # 1. 首先按原来的方式找出所有超过阈值的类别
        above_threshold = (Q_u > self.tau_c.unsqueeze(0)) # [batch_size, num_classes] 布尔类型

        # 2. 创建全零的伪标签矩阵
        P_u = torch.zeros_like(Q_u, device=self.device)
        
        # 3. 对于每个样本，如果有超过阈值的类别，只选择其中概率最高的那个
        for i in range(Q_u.shape[0]):
            # 获取当前样本超过阈值的类别索引
            valid_classes = torch.where(above_threshold[i])[0]
            
            if len(valid_classes) > 0:
                # 只有至少有一个类别超过阈值时才处理
                if len(valid_classes) == 1:
                    # 只有一个超过阈值的类别，直接设为1
                    P_u[i, valid_classes[0]] = 1.0
                else:
                    # 有多个超过阈值的类别，找出概率最高的那个
                    valid_probs = Q_u[i, valid_classes]
                    max_prob_idx = valid_classes[torch.argmax(valid_probs)]
                    P_u[i, max_prob_idx] = 1.0
        
        # 4. 创建掩码：有任何一个类别被标记为1的样本
        mask = (torch.sum(P_u, dim=1) > 0).float()
    
        return P_u, mask

    def _compute_losses(self, Y_l, L_l, Y_u, P_u, mask):
        """
        计算总损失 (内部方法)

        参数:
            Y_l: 有标签数据的预测 logits [batch_size_l, num_classes]
            L_l: 有标签数据的真实标签 [batch_size_l] (索引)
            Y_u: 无标签数据的预测 logits [batch_size_u, num_classes]
            P_u: 无标签数据的伪标签 [batch_size_u, num_classes] (0.0或1.0)
            mask: 有效伪标签的掩码 [batch_size_u] (0.0或1.0)

        返回:
            total_loss: 总损失
            sup_loss: 监督损失
            pseudo_loss: 伪标签损失
        """
        # 监督损失（交叉熵）
        # 确保标签不是-1 (数据加载器可能为无标签数据或无效数据设置-1)
        valid_labeled_indices = L_l != -1
        sup_loss = torch.tensor(0.0, device=self.device)
        if torch.any(valid_labeled_indices):
            criterion = nn.CrossEntropyLoss()
            sup_loss = criterion(Y_l[valid_labeled_indices], L_l[valid_labeled_indices])

        # 伪标签损失（带掩码的二元交叉熵）
        pseudo_loss = torch.tensor(0.0, device=self.device)
        num_valid_pseudo_samples = torch.sum(mask)

        if num_valid_pseudo_samples > 0 and Y_u.shape[0] > 0:
            # 计算每个样本的BCE损失 (对每个类别计算然后平均)
            # reduction='none' 返回 [batch_size_u, num_classes]
            pseudo_criterion = nn.BCEWithLogitsLoss(reduction='none')
            sample_pseudo_loss_per_class = pseudo_criterion(Y_u, P_u)

            # 平均每个样本的类别损失 [batch_size_u]
            sample_pseudo_loss = torch.mean(sample_pseudo_loss_per_class, dim=1)

            # 应用掩码并计算有效样本的平均损失
            # 只对 mask > 0 的样本计算损失
            pseudo_loss = torch.sum(sample_pseudo_loss * mask) / (num_valid_pseudo_samples + 1e-8)
        elif Y_u.shape[0] == 0:
            # 如果没有无标签数据批次，伪标签损失为0
             pseudo_loss = torch.tensor(0.0, device=self.device)


        # 总损失
        # 可以根据需要添加权重，例如 self.segment_loss_weight
        # total_loss = sup_loss + self.segment_loss_weight * pseudo_loss
        total_loss = sup_loss + pseudo_loss # 按照 markdown 中的简单相加

        return total_loss, sup_loss, pseudo_loss

    def _update_labeled_dataset(self, labeled_dataset, unlabeled_dataset, unlabeled_loader):
        """
        使用高置信度伪标签更新有标签数据集 (内部方法)

        参数:
            labeled_dataset: 当前的有标签数据集对象
            unlabeled_dataset: 无标签数据集对象
            unlabeled_loader: 无标签数据加载器

        返回:
            new_labeled_dataset: 更新后的有标签数据集对象 (可能是同一个对象被修改)
            selected_indices: 被选中的无标签数据索引 (在原始无标签集中的索引)
        """
        logger.info("开始更新有标签数据集...")
        self.model.eval() # 设置为评估模式
        self.graph_model.eval()

        all_confidences = []
        all_pseudo_labels = []
        all_original_indices = [] # 存储在unlabeled_dataset中的原始索引

        with torch.no_grad():
            # 使用当前的 tau_c (在训练循环中更新)
            # current_tau_c = self.tau_c.to(self.device) # 确保在正确设备

            # 遍历无标签数据集
            progress_bar = tqdm(unlabeled_loader, desc="Generating Pseudo Labels for Dataset Update")
            for batch in progress_bar:
                # 提取数据并移动到设备
                segment_features = batch['segment_features'].to(self.device)
                segment_attention_mask = batch['segment_attention_mask'].to(self.device)
                speakers = batch['speakers']
                input_ids = batch['input_ids'].to(self.device)
                input_features = batch['input_features'].to(self.device) if batch['input_features'] is not None else None
                # --- 添加类型转换 ---
                if input_features is not None and hasattr(self.model.thinker, 'dtype') and self.model.thinker.dtype == torch.bfloat16:
                    input_features = input_features.to(torch.bfloat16)
                    
                # --- 转换 segment_features ---
                if segment_features is not None and hasattr(self.graph_model, 'dtype') and self.graph_model.dtype == torch.bfloat16:
                    segment_features = segment_features.to(torch.bfloat16)

                # print(f"input_features: {input_features.shape}") # 保持你添加的打印语句（如果需要）
                feature_attention_mask = batch['feature_attention_mask'].to(self.device) if batch['feature_attention_mask'] is not None else None
                attention_mask = batch['attention_mask'].to(self.device) if batch['attention_mask'] is not None else None
                original_indices = batch['index'] # 获取样本在原始数据集中的索引

                # 处理图模型
                graph_feature = None
                try:
                    graph_embedding = self.graph_model(segment_features, speakers, attention_masks=segment_attention_mask)
                    graph_feature = graph_embedding
                except Exception as e:
                    logger.warning(f"图网络处理失败 (评估阶段): {e}")
                    # 即使图处理失败，也继续，但可能影响模型性能

                # 构造模型输入
                inputs = {
                    "input_ids": input_ids,
                    "input_features": input_features,
                    "feature_attention_mask": feature_attention_mask,
                    "attention_mask": attention_mask,
                    "graph_audio_features": graph_feature
                }
                # **重要假设**: outputs.logits 是 [batch_size, num_classes] 分类 logits
                outputs = self.model.thinker(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1) # 获取概率 [batch_size, num_classes]

                # 生成伪标签 (使用当前阈值)
                P_u_batch, mask_batch = self._generate_pseudo_labels(probs) 
                # P_u_batch 形状: [batch_size, num_classes], 已经是 one-hot 或全零
                # mask_batch 形状: [batch_size], 标记哪些样本有伪标签 (值为 1.0)

                # 计算置信度（最大预测概率）
                max_probs, _ = torch.max(probs, dim=1) # [batch_size]

                # 直接使用 _generate_pseudo_labels 返回的 mask_batch 作为 valid_mask
                valid_mask = mask_batch.bool() # 转换为布尔类型以便索引或迭代

                for i, is_valid in enumerate(valid_mask):
                    if is_valid: # 检查样本是否有伪标签
                        all_confidences.append(max_probs[i].item())
                        
                        # 对 P_u_batch (one-hot 伪标签) 取 argmax 获取类别索引
                        pseudo_label_index = torch.argmax(P_u_batch[i]).cpu().item()
                        
                        all_pseudo_labels.append(pseudo_label_index) # 存储类别索引
                        all_original_indices.append(original_indices[i]) # 存储原始索引

        # 根据置信度排序并选择top-k%
        selected_indices_in_unlabeled = [] # 存储被选中的样本在unlabeled_dataset中的索引
        selected_pseudo_labels_values = [] # 存储对应的伪标签值
        num_valid_pseudo = len(all_confidences)

        if num_valid_pseudo > 0:
            sorted_indices = np.argsort(all_confidences)[::-1]  # 按置信度从高到低排序

            num_to_select = max(1, int(num_valid_pseudo * self.top_k_percent))
            selected_indices_in_confidence_list = sorted_indices[:num_to_select] # 获取置信度列表中的索引

            # 从原始索引和伪标签列表中提取选中的项
            selected_indices_in_unlabeled = [all_original_indices[i] for i in selected_indices_in_confidence_list]
            selected_pseudo_labels_values = [all_pseudo_labels[i] for i in selected_indices_in_confidence_list]

            logger.info(f"从 {num_valid_pseudo} 个有效伪标签样本中，根据置信度选择了最高的 {len(selected_indices_in_unlabeled)} 个样本 (top {self.top_k_percent * 100:.1f}%)")

            # 更新有标签数据集
            # **注意**: 这需要 AudioSegmentDataset 支持添加数据，或者我们创建一个新的数据集
            # 假设 labeled_dataset 有一个 add_pseudo_labeled_data 方法
            try:
                labeled_dataset.add_pseudo_labeled_data(unlabeled_dataset, selected_indices_in_unlabeled, selected_pseudo_labels_values)
                logger.info(f"成功将 {len(selected_indices_in_unlabeled)} 个伪标签样本添加到有标签数据集中。")
            except AttributeError:
                 logger.error("Labeled dataset object does not have 'add_pseudo_labeled_data' method. Dataset update skipped.")
                 # 如果没有该方法，则无法更新数据集
                 selected_indices_in_unlabeled = [] # 清空选择，因为无法添加
            except Exception as e:
                 logger.error(f"添加伪标签数据时出错: {e}")
                 selected_indices_in_unlabeled = [] # 清空选择

        else:
            logger.warning("在无标签数据集中没有找到有效的、满足阈值的伪标签样本。")

        # 恢复训练模式
        self.model.train()
        self.graph_model.train()

        # 返回更新后的数据集对象和选中的索引
        return labeled_dataset, selected_indices_in_unlabeled
    
    def get_class_probabilities(self, outputs):
        ## 获取最后一个位置4个标签各自的概率
        logits = outputs.logits  # 形状为[batch_size, sequence_length, vocab_size]
        last_token_logits = logits[:, -1, :]  # 取最后一个位置的预测，形状为[batch_size, vocab_size]
        #  获取A、B、C、D选项对应的token_id
        option_tokens = [
            self.processor.tokenizer.convert_tokens_to_ids("A"),
            self.processor.tokenizer.convert_tokens_to_ids("B"),
            self.processor.tokenizer.convert_tokens_to_ids("C"),
            self.processor.tokenizer.convert_tokens_to_ids("D")
        ]

        #  提取这四个选项的logits
        batch_size = last_token_logits.shape[0]
        num_classes = len(option_tokens)
        class_logits = torch.zeros((batch_size, num_classes), device=last_token_logits.device)

        for i, token_id in enumerate(option_tokens):
            class_logits[:, i] = last_token_logits[:, token_id]

        #  应用softmax获取概率
        probability_scores = F.softmax(class_logits, dim=1)  # [batch_size, num_classes]

        # 如果需要，可以将概率转回CPU并转为numpy数组
        # probability_scores_np = probability_scores.cpu().numpy()

        # 返回形状为[batch_size, num_classes]的概率分布
        probability_scores_cpu = probability_scores.cpu() # Move to CPU before returning if possible, or just return GPU tensor
        del logits, last_token_logits, class_logits # Explicitly delete intermediate tensors
        return probability_scores_cpu # Return GPU tensor directly
    
    def train(
        self,
        train_data_path,
        num_epochs=5,
        gradient_accumulation_steps=4,
        log_interval=10,
        save_interval=1000,
        use_wandb=False,
        batch_size=1,
        use_balanced_sampling=True
    ):
        """
        训练模型，支持半监督学习

        参数:
            train_data_path: 有标签训练数据CSV文件所在的目录
            num_epochs: 训练轮数
            gradient_accumulation_steps: 梯度累积步数
            log_interval: 日志记录间隔
            save_interval: 保存间隔
            use_wandb: 是否使用wandb记录实验
            num_segments: 每个音频切分的片段数量，None表示随机
            batch_size: 批次大小
            use_balanced_sampling: 是否使用平衡采样
        """
        if self.use_semi_supervised:
             logger.info("启用半监督学习模式。")

        # 初始化wandb
        if use_wandb:
            wandb_config = {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                # "segment_loss_weight": self.segment_loss_weight, # 暂时不用
                "num_epochs": num_epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "batch_size": batch_size,
                "use_lora": self.use_lora,
                "lora_r": self.lora_r if self.use_lora else None,
                "lora_alpha": self.lora_alpha if self.use_lora else None,
                "lora_dropout": self.lora_dropout if self.use_lora else None,
                "use_semi_supervised": self.use_semi_supervised,
                
            }
            if self.use_semi_supervised:
                 wandb_config.update({
                    "lambda_ema": self.lambda_ema,
                    "initial_tau": self.initial_tau,
                    "top_k_percent": self.top_k_percent,
                    "update_dataset_freq": self.update_dataset_freq,
                 })
            wandb.init(project="audio-llm-telemarketing", config=wandb_config)

        # 创建训练数据集和数据加载器
        train_labels_file = 'train.csv'
        logger.info(f"使用训练文件: {os.path.join(train_data_path, train_labels_file)}")
        train_config = DataLoaderConfig(
            data_path=train_data_path,
            labels_file=train_labels_file,
            cache_dir='features_cache', 
            batch_size=batch_size,
            balance_labels=use_balanced_sampling,
            model_path=self.processor.name_or_path if hasattr(self.processor, 'name_or_path') else None
        )
        train_dataset = AudioSegmentDataset(
            data_path=train_config.data_path,
            model_path=train_config.model_path,
            data_processor=self.processor,
            labels_file=train_config.labels_file,
            cache_dir=train_config.cache_dir,
            )
        
        # 计算有标签和无标签数据的数量
        labeled_indices = []
        unlabeled_indices = []
        for i, item in enumerate(train_dataset.data):
            if item.get('label') is not None and str(item.get('label')).strip():
                labeled_indices.append(i)
            else:
                unlabeled_indices.append(i)
        
        logger.info(f"训练数据集共 {len(train_dataset)} 个样本")
        logger.info(f"其中有标签样本: {len(labeled_indices)} 个")
        logger.info(f"无标签样本: {len(unlabeled_indices)} 个")
        
        train_sampler = None
        if train_config.balance_labels:
            train_sampler = LabelBalancedSampler(dataset=train_dataset, batch_size=train_config.batch_size)
            logger.info(f"已启用标签均衡采样")
        train_loader = DataLoader(
            train_dataset, batch_size=train_config.batch_size, sampler=train_sampler,
            collate_fn=train_dataset.collate_fn, shuffle=(train_sampler is None) # 如果不用sampler则shuffle
        )

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 训练循环
        global_step = 0
        total_steps = len(train_loader) * num_epochs

        # --- 在训练开始前重置/初始化阈值 ---
        if self.use_semi_supervised:
             self.tau = torch.tensor(self.initial_tau, device=self.device)
             self.p_tilde = torch.ones(self.num_classes, device=self.device) / self.num_classes
             self.tau_c = self.tau * (self.p_tilde / torch.max(self.p_tilde))

        for epoch in range(num_epochs):
            self.model.train()
            self.graph_model.train()

            epoch_loss = 0
            epoch_sup_loss = 0
            epoch_pseudo_loss = 0
            num_batches = 0

            # 处理有标签数据
            labeled_progress_bar = tqdm(
                [train_loader.dataset[i] for i in labeled_indices], 
                desc=f"Epoch {epoch+1}/{num_epochs} (有标签数据)"
            )
            
            # 收集有标签数据的批次
            labeled_batches = []
            batch_items = []
            for item in labeled_progress_bar:
                batch_items.append(item)
                if len(batch_items) == batch_size:
                    labeled_batches.append(train_dataset.collate_fn(batch_items))
                    batch_items = []
            if batch_items:  # 处理最后一个不完整的批次
                labeled_batches.append(train_dataset.collate_fn(batch_items))
            
            # 训练有标签数据
            logger.info(f"处理 {len(labeled_batches)} 个有标签数据批次")
            for batch_idx, labeled_batch in enumerate(labeled_batches):
                num_batches += 1
                
                # 处理有标签数据
                labeled_inputs_dict = self._prepare_batch_inputs(labeled_batch)
                labeled_labels = torch.tensor([l for l in labeled_batch['label'] if l is not None and str(l).strip()], device=self.device, dtype=torch.bfloat16)
                
                # 有标签数据前向传播
                outputs_l = self.model.thinker(**labeled_inputs_dict)
                Y_l = self.get_class_probabilities(outputs_l)
                
                # 初始化无标签数据的空张量
                Y_u = torch.empty(0, self.num_classes, device=self.device)
                P_u = torch.empty(0, self.num_classes, device=self.device)
                mask = torch.empty(0, device=self.device)
                
                # 计算监督损失
                loss, sup_loss, pseudo_loss = self._compute_losses(Y_l, labeled_labels, Y_u, P_u, mask)
                
                # 累积梯度
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_sup_loss += sup_loss.item()
                
                # Explicitly delete tensors no longer needed for this batch
                del labeled_inputs_dict, outputs_l, Y_l, loss, sup_loss, pseudo_loss
                # Optional: If memory pressure is extreme, uncommenting this might help, but can slow down training.
                # torch.cuda.empty_cache()

                # 梯度累积和优化器步骤
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx + 1 == len(labeled_batches):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # 记录日志和保存检查点
                    if global_step % log_interval == 0 and use_wandb:
                        avg_loss = epoch_loss / num_batches
                        avg_sup_loss = epoch_sup_loss / num_batches
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/supervised_loss": avg_sup_loss,
                            "global_step": global_step,
                            "progress": global_step / total_steps
                        })
                    
                    if global_step % save_interval == 0:
                        self.save_model(os.path.join(self.output_dir, f"checkpoint-{global_step}"))
            
            # 处理无标签数据（如果启用半监督学习）
            if self.use_semi_supervised and unlabeled_indices:
                unlabeled_progress_bar = tqdm(
                    [train_loader.dataset[i] for i in unlabeled_indices], 
                    desc=f"Epoch {epoch+1}/{num_epochs} (无标签数据)"
                )
                
                # 收集无标签数据的批次
                unlabeled_batches = []
                batch_items = []
                for item in unlabeled_progress_bar:
                    batch_items.append(item)
                    if len(batch_items) == batch_size:
                        unlabeled_batches.append(train_dataset.collate_fn(batch_items))
                        batch_items = []
                if batch_items:  # 处理最后一个不完整的批次
                    unlabeled_batches.append(train_dataset.collate_fn(batch_items))
                
                # 训练无标签数据
                logger.info(f"处理 {len(unlabeled_batches)} 个无标签数据批次")
                for batch_idx, unlabeled_batch in enumerate(unlabeled_batches):
                    num_batches += 1
                    
                    unlabeled_inputs_dict = self._prepare_batch_inputs(unlabeled_batch)
                    
                    # 1. 前向传播获取伪标签概率 (无梯度)
                    with torch.no_grad():
                        outputs_u_pseudo = self.model.thinker(**unlabeled_inputs_dict)
                        Q_u = self.get_class_probabilities(outputs_u_pseudo)
                    
                    # 2. 更新自适应阈值
                    self._update_adaptive_thresholds(Q_u)
                    
                    # 3. 生成伪标签
                    P_u, mask = self._generate_pseudo_labels(Q_u)
                    
                    # 4. 再次前向传播计算损失 (有梯度)
                    outputs_u = self.model.thinker(**unlabeled_inputs_dict)
                    Y_u = self.get_class_probabilities(outputs_u)
                    
                    # 初始化有标签数据的空张量和标签
                    Y_l = torch.empty(0, self.num_classes, device=self.device)
                    L_l = torch.empty(0, dtype=torch.long, device=self.device)
                    
                    # 计算伪标签损失
                    loss, sup_loss, pseudo_loss = self._compute_losses(Y_l, L_l, Y_u, P_u, mask)
                    
                    # 累积梯度
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    epoch_pseudo_loss += pseudo_loss.item()

                    # Explicitly delete tensors no longer needed for this batch
                    del unlabeled_inputs_dict, outputs_u_pseudo, Q_u, P_u, mask, outputs_u, Y_u, Y_l, L_l, loss, sup_loss, pseudo_loss
                    # Optional: If memory pressure is extreme, uncommenting this might help, but can slow down training.
                    # torch.cuda.empty_cache()

                    # 梯度累积和优化器步骤
                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx + 1 == len(unlabeled_batches):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                        
                        # 记录日志和保存检查点
                        if global_step % log_interval == 0 and use_wandb:
                            avg_loss = epoch_loss / num_batches
                            avg_pseudo_loss = epoch_pseudo_loss / max(1, len(unlabeled_batches))
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/pseudo_loss": avg_pseudo_loss,
                                "train/global_threshold": self.tau.item(),
                                "global_step": global_step, 
                                "progress": global_step / total_steps
                            })
                        
                        if global_step % save_interval == 0:
                            self.save_model(os.path.join(self.output_dir, f"checkpoint-{global_step}"))

            # --- Epoch 结束 ---
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_sup_loss = epoch_sup_loss / max(1, len(labeled_batches))
            avg_epoch_pseudo_loss = epoch_pseudo_loss / max(1, len(unlabeled_indices) // batch_size)

            logger.info(f"Epoch {epoch+1}/{num_epochs} completed:")
            logger.info(f"  Average Total Loss: {avg_epoch_loss:.4f}")
            logger.info(f"  Average Supervised Loss: {avg_epoch_sup_loss:.4f}")
            if self.use_semi_supervised and unlabeled_indices:
                logger.info(f"  Average Pseudo Label Loss: {avg_epoch_pseudo_loss:.4f}")
                logger.info(f"  End-of-Epoch Global Threshold: {self.tau.item():.4f}")

            if use_wandb:
                log_data = {
                    "epoch": epoch + 1,
                    "epoch/loss": avg_epoch_loss,
                    "epoch/supervised_loss": avg_epoch_sup_loss,
                }
                if self.use_semi_supervised and unlabeled_indices:
                     log_data["epoch/pseudo_loss"] = avg_epoch_pseudo_loss
                     log_data["epoch/global_threshold"] = self.tau.item()
                wandb.log(log_data)

            # --- 更新数据集中的伪标签 --- 
            if self.use_semi_supervised and unlabeled_indices and \
               (epoch + 1) % self.update_dataset_freq == 0 and epoch < num_epochs - 1:
                # 更新索引列表 (将一些无标签数据转移到有标签数据)
                logger.info(f"开始更新伪标签样本...")
                
                # 遍历所有无标签数据，找出满足伪标签条件的数据
                self.model.eval()
                self.graph_model.eval()
                
                all_confidences = []
                all_pseudo_labels = []
                all_original_indices = []
                
                with torch.no_grad():
                    # 依次处理每个无标签样本
                    for i, idx in enumerate(unlabeled_indices):
                        item = train_dataset.data[idx]
                        # 构建单个样本的批次
                        batch = train_dataset.collate_fn([item])
                        inputs_dict = self._prepare_batch_inputs(batch)
                        
                        # 获取预测
                        outputs = self.model.thinker(**inputs_dict)
                        probs = self.get_class_probabilities(outputs)
                        
                        # 生成伪标签
                        P_u_batch, mask_batch = self._generate_pseudo_labels(probs)
                        
                        # 计算置信度
                        max_probs, _ = torch.max(probs, dim=1)
                        
                        # 检查是否有有效伪标签
                        valid_mask = mask_batch.bool()
                        for j, is_valid in enumerate(valid_mask):
                            if is_valid:
                                all_confidences.append(max_probs[j].item())
                                pseudo_label_index = torch.argmax(P_u_batch[j]).cpu().item()
                                all_pseudo_labels.append(pseudo_label_index)
                                all_original_indices.append(idx)

                        # Explicitly delete tensors for this item/batch
                        del item, batch, inputs_dict, outputs, probs, P_u_batch, mask_batch, max_probs, valid_mask
                        # Optional: torch.cuda.empty_cache()
                
                # 选择最高置信度的样本
                num_valid_pseudo = len(all_confidences)
                if num_valid_pseudo > 0:
                    sorted_indices = np.argsort(all_confidences)[::-1]
                    num_to_select = max(1, int(num_valid_pseudo * self.top_k_percent))
                    selected_indices = sorted_indices[:num_to_select]
                    
                    # 更新数据集的伪标签
                    newly_labeled = []
                    for i in selected_indices:
                        orig_idx = all_original_indices[i]
                        pseudo_label = all_pseudo_labels[i]
                        # 更新数据集中的标签
                        train_dataset.data[orig_idx]['label'] = pseudo_label
                        newly_labeled.append(orig_idx)
                    
                    # 更新有标签和无标签索引列表
                    unlabeled_indices = [idx for idx in unlabeled_indices if idx not in newly_labeled]
                    labeled_indices.extend(newly_labeled)
                    
                    logger.info(f"从 {num_valid_pseudo} 个有效伪标签样本中，选择了置信度最高的 {len(newly_labeled)} 个样本作为新的有标签数据")
                    logger.info(f"现有标签样本: {len(labeled_indices)} 个, 无标签样本: {len(unlabeled_indices)} 个")
                else:
                    logger.warning("没有找到满足伪标签条件的无标签样本")
                
                # 恢复训练模式
                self.model.train()
                self.graph_model.train()

            # 保存每个epoch的模型
            self.save_model(os.path.join(self.output_dir, f"epoch-{epoch+1}"))

        # 保存最终模型
        self.save_model(os.path.join(self.output_dir, "final-model"))

        if use_wandb:
            wandb.finish()

    def _prepare_batch_inputs(self, batch):
        """将批次数据移动到设备并准备模型输入字典 (内部方法)"""


        speakers = batch['speakers']
        # Move to device
        segment_features = batch['segment_features'].to(self.device) if batch['segment_features'] is not None else None
        segment_attention_mask = batch['segment_attention_mask'].to(self.device) if batch['segment_attention_mask'] is not None else None
        input_ids = batch['input_ids'].to(self.device) if batch['input_ids'] is not None else None
        input_features = batch['input_features'].to(self.device) if batch['input_features'] is not None else None
        # --- 添加类型转换 ---
        if input_features is not None and hasattr(self.model.thinker, 'dtype') and self.model.thinker.dtype == torch.bfloat16:
            input_features = input_features.to(torch.bfloat16)
            
        # --- 转换 segment_features ---
        if segment_features is not None and hasattr(self.graph_model, 'dtype') and self.graph_model.dtype == torch.bfloat16:
            segment_features = segment_features.to(torch.bfloat16)

        # print(f"input_features: {input_features.shape}") # 保持你添加的打印语句（如果需要）
        feature_attention_mask = batch['feature_attention_mask'].to(self.device) if batch['feature_attention_mask'] is not None else None
        attention_mask = batch['attention_mask'].to(self.device) if batch['attention_mask'] is not None else None


        # 使用图网络处理segment_features
        graph_feature = None
        if segment_features is not None and segment_features.nelement() > 0: # 检查非空
             try:
                 graph_embedding = self.graph_model(
                     segment_features,
                     speakers,
                     attention_masks=segment_attention_mask
                 )
                 graph_feature = graph_embedding # 形状: [batch, output_dim]
             except Exception as e:
                 logger.warning(f"图网络处理失败: {e}")
                 # 可以选择填充零张量或根据需要处理
                 # graph_feature = torch.zeros(segment_features.shape[0], self.graph_model.output_dim, device=self.device)


        # 构造模型输入字典
        inputs = {
            "input_ids": input_ids,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "attention_mask": attention_mask,
            "graph_audio_features": graph_feature
        }
        # 移除值为 None 的键
        inputs = {k: v for k, v in inputs.items() if v is not None}
        return inputs

    def test(self, test_loader):
        """
        在测试集上评估最终模型 (基本不变, 确保使用 _prepare_batch_inputs)

        参数:
            test_loader: 测试数据加载器

        返回:
            测试结果字典
        """
        logger.info("Starting final evaluation on the test set...")
        self.model.eval()
        self.graph_model.eval()

        test_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                num_batches += 1

                inputs_dict = self._prepare_batch_inputs(batch)
                labels = batch['label'].to(self.device) # 获取标签

                try:
                    outputs = self.model.thinker(**inputs_dict, labels=labels if torch.any(labels != -1) else None) # 只有在有有效标签时传递

                    if outputs.loss is not None:
                        test_loss += outputs.loss.item()
                    elif hasattr(outputs, 'logits'): # 手动计算损失（如果需要）
                         logits = outputs.logits
                         valid_indices = labels != -1
                         if torch.any(valid_indices):
                             loss = F.cross_entropy(logits[valid_indices], labels[valid_indices])
                             test_loss += loss.item()

                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                except Exception as e:
                    logger.error(f"测试时处理批次出错: {e}")
                    import traceback
                    traceback.print_exc()
                    batch_size = labels.size(0) if labels is not None else 1
                    all_preds.extend([-1] * batch_size)
                    all_labels.extend(labels.cpu().numpy() if labels is not None else [-1] * batch_size)
                    continue

        # 计算平均损失 (仅基于有标签的样本)
        valid_loss_count = sum(1 for label in all_labels if label != -1)
        avg_loss = test_loss / max(valid_loss_count, 1) # 除以有效样本数

        results = {"test_loss": avg_loss}

        if all_preds and all_labels and len(all_preds) == len(all_labels):
            valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
            if valid_indices:
                valid_preds = [all_preds[i] for i in valid_indices]
                valid_labels = [all_labels[i] for i in valid_indices]

                if len(valid_labels) > 0:
                    accuracy = accuracy_score(valid_labels, valid_preds)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        valid_labels, valid_preds, average='weighted', zero_division=0
                    )
                    results.update({"test_accuracy": accuracy, "test_precision": precision, "test_recall": recall, "test_f1": f1})
                else:
                    logger.warning("测试集中没有有效的标签可用于计算指标")
            else:
                 logger.warning("测试集中没有有效的标签可用于计算指标")

        logger.info("Test set evaluation finished.")
        logger.info(f"Test Results: {results}")
        return results

    def save_model(self, output_path):
        """
        保存模型 (添加半监督配置)

        参数:
            output_path: 输出路径
        """
        os.makedirs(output_path, exist_ok=True)

        if self.use_lora:
            self.model.save_pretrained(output_path)
            logger.info(f"LoRA适配器保存到: {output_path}")
        else:
            self.model.save_pretrained(output_path)

        self.processor.save_pretrained(output_path)
        torch.save(self.graph_model.state_dict(), os.path.join(output_path, "graph_model.pt"))

        # 保存训练配置 (包含半监督参数)
        trainer_config_data = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            # "segment_loss_weight": self.segment_loss_weight, # 如果未来使用，取消注释
            "num_classes": self.num_classes,
            "use_lora": self.use_lora,
            "lora_r": getattr(self, "lora_r", 8) if self.use_lora else None,
            "lora_alpha": getattr(self, "lora_alpha", 16) if self.use_lora else None,
            "lora_dropout": getattr(self, "lora_dropout", 0.05) if self.use_lora else None,
            # 保存半监督参数
            "use_semi_supervised": self.use_semi_supervised,
            "lambda_ema": self.lambda_ema,
            "initial_tau": self.initial_tau,
            "top_k_percent": self.top_k_percent,
            "update_dataset_freq": self.update_dataset_freq,
            # 保存图模型配置 (如果需要恢复)
            # "graph_config": self.graph_model.config # 假设图模型有config属性
        }
        with open(os.path.join(output_path, "trainer_config.json"), "w") as f:
            import json
            json.dump(trainer_config_data, f, indent=4)

        logger.info(f"模型和配置保存到: {output_path}")

    def load_model(self, model_path):
        """
        加载模型 (读取半监督配置)

        参数:
            model_path: 模型路径
        """
        config_path = os.path.join(model_path, "trainer_config.json")
        is_lora_model = False
        loaded_config = {}

        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                is_lora_model = loaded_config.get('use_lora', False)
                logger.info(f"检测到配置文件，LoRA模式: {is_lora_model}")
                # 更新训练器状态 (例如 LoRA, 半监督设置等)
                self.use_lora = is_lora_model
                self.use_semi_supervised = loaded_config.get('use_semi_supervised', False)
                self.lambda_ema = loaded_config.get('lambda_ema', 0.9)
                self.initial_tau = loaded_config.get('initial_tau', 0.8)
                self.top_k_percent = loaded_config.get('top_k_percent', 0.1)
                self.update_dataset_freq = loaded_config.get('update_dataset_freq', 1)
                self.num_classes = loaded_config.get('num_classes', 4) # 恢复 num_classes
                logger.info(f"半监督模式: {self.use_semi_supervised}")

            except Exception as e:
                logger.error(f"读取配置文件 {config_path} 时出错: {e}")
                logger.warning("将尝试使用默认方式加载模型")
        else:
            logger.warning(f"配置文件 {config_path} 未找到，无法恢复训练器状态")


        # --- 加载模型权重 ---
        try:
            # 根据配置文件或检测到的LoRA状态加载
            if self.use_lora: # 使用更新后的 self.use_lora
                logger.info(f"以LoRA模式加载模型适配器: {model_path}")
                # 确保基础模型已准备好接收适配器
                if not hasattr(self.model, 'load_adapter'):
                     logger.warning("当前模型不是PEFT模型，将尝试转换...")
                     # 需要原始的 LoRA 配置来重建，尝试从加载的配置中获取
                     lora_r = loaded_config.get('lora_r', 8)
                     lora_alpha = loaded_config.get('lora_alpha', 16)
                     lora_dropout = loaded_config.get('lora_dropout', 0.05)
                     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] # 或从配置加载
                     lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
                     # 假设 self.model 是基础模型
                     self.model = get_peft_model(self.model, lora_config)

                self.model.load_adapter(model_path) # 加载适配器权重
                logger.info(f"LoRA适配器从 {model_path} 加载成功")
            else:
                # 加载完整模型
                logger.info(f"以全参数模式加载模型: {model_path}")
                # 创建一个临时的基础模型实例来加载权重，然后赋值回 self.model
                # 这假设 self.model 当前是基础模型或可以被替换
                base_model_class = type(self.model.base_model.model) if hasattr(self.model, 'base_model') else type(self.model)
                if torch.cuda.is_available():
                    temp_model = base_model_class.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
                    )
                else:
                    temp_model = base_model_class.from_pretrained(
                        model_path, torch_dtype=torch.float32
                    )
                self.model = temp_model # 替换当前模型
                logger.info(f"完整模型从 {model_path} 加载成功")

        except Exception as e:
            logger.error(f"加载模型权重失败: {e}")
            import traceback
            traceback.print_exc()

        # --- 加载图模型 ---
        graph_model_path = os.path.join(model_path, "graph_model.pt")
        if os.path.exists(graph_model_path):
            try:
                # 确保图模型结构匹配 (如果配置变化需要重新初始化)
                self.graph_model.load_state_dict(torch.load(graph_model_path, map_location=self.device))
                self.graph_model.to(self.device)
                logger.info(f"图模型从 {graph_model_path} 加载成功")
            except Exception as e:
                 logger.error(f"加载图模型状态字典失败: {e}")
        else:
            logger.warning(f"图模型文件 {graph_model_path} 不存在，未加载图模型权重")

        # # --- 加载处理器 ---
        # try:
        #     self.processor = Qwen2_5OmniLightProcessor.from_pretrained(model_path) # 使用具体类
        #     logger.info(f"处理器从 {model_path} 加载成功")
        # except Exception as e:
        #     logger.error(f"加载处理器失败: {e}")

        # --- 重新初始化优化器 (加载后可能需要) ---
        self._init_optimizer()
        logger.info("优化器已重新初始化。")


def main(
    batch_size=1,
    use_balanced_sampling=True,
    use_wandb=True,
    use_lora=True,
    # --- 添加半监督控制参数 ---
    use_semi_supervised=False,
    lambda_ema=0.9,
    initial_tau=0.8,
    top_k_percent=0.1,
    update_dataset_freq=1
    ):
    """主函数，支持半监督学习"""
    # 配置参数
    model_path = "/data/shared/Qwen/models/Qwen2.5-Omni-7B"
    output_dir = "./outputs_semi" if use_semi_supervised else "./outputs" # 根据模式选择不同输出目录
    data_dir = "./data" # 包含 train.csv, audio/, segments/ 的目录
    train_data_path = data_dir
    test_data_path = data_dir # 测试集路径

    # 如果启用半监督，检查无标签数据路径
    effective_use_semi = use_semi_supervised
    if not use_semi_supervised:
        logger.info("半监督模式未启用。")

    # 加载processor 和 config
    try:
        processor = Qwen2_5OmniLightProcessor.from_pretrained(model_path)
        llm_config = AutoConfig.from_pretrained(model_path)
         # **重要**: 设置分类任务的类别数量，这需要根据你的任务确定
        num_classes = 4 # 假设是4分类，需要根据实际任务修改！
        # 如果模型本身不是分类模型，可能需要修改配置添加分类头
        # llm_config.num_labels = num_classes # 例如
        logger.info(f"分类任务类别数量: {num_classes}")

    except Exception as e:
        logger.error(f"加载 Processor 或 Config 失败: {e}")
        return

    # 加载模型
    if torch.cuda.is_available():
        device = "cuda:0"
        logger.info("CUDA可用，将使用GPU加载模型")
        try:
            model = Qwen2_5OmniTextOnlyModel.from_pretrained(
                model_path,
                config=llm_config, # 应用可能修改过的配置
                device_map="cuda:0",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
            print(f"the model dtype is {model.dtype}")
        except Exception as e:
             logger.error(f"加载模型到GPU失败: {e}")
             return
    else:
        logger.error("CUDA不可用, 终止训练。半监督和LoRA通常需要GPU。")
        return

    # 获取LLM的隐藏层大小
    llm_hidden_size = 3584 
    logger.info(f"LLM hidden size: {llm_hidden_size}")

    # 创建图模型配置
    graph_config_dict = {
        "token_embedding_dim": llm_hidden_size, # 使用LLM的隐藏层维度
        "output_dim": llm_hidden_size,          # 输出维度与输入保持一致
        "num_heads": 8,
        "speaker_embedding_dim": 32,
        "num_speakers": None, # 让模型动态处理
        "num_layers": 2,
        "dropout": 0.1,
        "similarity_threshold": 0.7,
        "context_window_size": 5,
        "dtype": torch.bfloat16 # 使用与主模型相同的精度
    }
    logger.info("图模型配置:")
    for key, value in graph_config_dict.items(): logger.info(f" {key}: {value}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # LoRA配置
    lora_r = 16 # 尝试更大的 r
    lora_alpha = 32

    if use_lora: logger.info(f"使用LoRA微调，参数: r={lora_r}, alpha={lora_alpha}")
    else: logger.info("使用全参数微调")

    # 初始化训练器
    trainer = AdaptiveThresholdTrainer(
        model=model,
        processor=processor,
        graph_config=graph_config_dict,
        output_dir=output_dir,
        device=device,
        learning_rate=1e-5 if use_lora else 5e-6, # LoRA可以用稍大学习率
        weight_decay=0.01,
        num_classes=num_classes, # 传递类别数
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        # 传递半监督参数
        use_semi_supervised=effective_use_semi,
        lambda_ema=lambda_ema,
        initial_tau=initial_tau,
        top_k_percent=top_k_percent,
        update_dataset_freq=update_dataset_freq
    )

    # 训练模型
    trainer.train(
        train_data_path=train_data_path,
        num_epochs=5, # 增加 epoch 数量
        gradient_accumulation_steps=8, # 增大累积步数以模拟更大batch size
        log_interval=10,
        save_interval=500, # 保存频率
        use_wandb=use_wandb,
        batch_size=batch_size,
        use_balanced_sampling=use_balanced_sampling
    )

    # --- 测试阶段 ---
    logger.info("训练完成。加载最终模型进行测试...")
    final_model_path = os.path.join(output_dir, "final-model")

    try:
        trainer.load_model(final_model_path)
        logger.info(f"最终模型从 {final_model_path} 加载成功。")
    except Exception as e:
        logger.error(f"加载最终模型 {final_model_path} 失败: {e}. 无法执行测试评估。")
        if use_wandb: wandb.finish()
        return # 无法加载模型，退出

    # 创建测试集数据加载器
    test_labels_file = 'test.csv'
    test_file_path = os.path.join(test_data_path, test_labels_file)
    logger.info(f"尝试加载测试文件: {test_file_path}")

    if not os.path.exists(test_file_path):
        logger.warning(f"测试标签文件 {test_file_path} 不存在，跳过测试评估。")
    else:
        test_config = DataLoaderConfig(
            data_path=test_data_path,
            labels_file=test_labels_file,
            cache_dir='features_cache', # 使用单独的测试缓存目录
            batch_size=batch_size, # 测试时batch size可以适当增大
            balance_labels=False,
            model_path=trainer.processor.name_or_path if hasattr(trainer.processor, 'name_or_path') else None
        )
        try:
            test_dataset = AudioSegmentDataset(
                data_path=test_config.data_path,
                model_path=test_config.model_path,
                data_processor=trainer.processor, # 使用加载后的processor
                labels_file=test_config.labels_file,
                cache_dir=test_config.cache_dir,
                num_classes=trainer.num_classes # 使用加载后的num_classes
            )
            test_loader = DataLoader(
                test_dataset, batch_size=test_config.batch_size,
                collate_fn=test_dataset.collate_fn
                # num_workers=test_config.num_workers # 根据需要设置 num_workers
            )
            logger.info(f"测试数据加载器创建成功，共 {len(test_dataset)} 个样本。")

            # 执行测试评估
            test_results = trainer.test(test_loader)

            # 记录测试结果 (WandB)
            if use_wandb and test_results:
                wandb_test_results = {f"test/{k.replace('test_', '')}": v for k, v in test_results.items()}
                wandb.log(wandb_test_results)

        except FileNotFoundError:
            logger.warning(f"测试标签文件 {test_labels_file} 存在但无法在 Dataset 中加载，跳过测试评估。")
        except Exception as e:
             logger.error(f"创建测试数据加载器或执行测试时出错: {e}")
             import traceback
             traceback.print_exc()
             logger.warning("跳过测试评估。")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # 示例：运行半监督学习
    main(
        use_lora=True,
        use_wandb=True,
        batch_size=1, # 根据你的GPU显存调整
        use_balanced_sampling=True,
        use_semi_supervised=True, # 启用半监督
        lambda_ema=0.95, # 调整EMA参数
        initial_tau=0.9, # 调整初始阈值
        top_k_percent=0.05, # 选择更少但更高质量的伪标签
        update_dataset_freq=2 # 每2个epoch更新一次数据集
    )
