import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from tqdm import tqdm
import logging
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import transforms
import random
import sys

from utils.graph import DialogueGraphModel
from utils.audio_augment import AudioAugmenter
from utils.dataloader import create_telemarketing_dataloader
from utils.model import AudioEncoder

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioDialogueDataset(Dataset):
    """音频对话数据集，用于加载和处理音频对话数据"""
    
    def __init__(self, data_path, tokenizer, max_length=512, labeled_ratio=0.3, total_batches=1000, preprocessed_path=None):
        """
        初始化数据集
        
        参数:
            data_path: 数据路径
            tokenizer: 分词器
            max_length: 最大序列长度
            labeled_ratio: 有标签数据的比例
            total_batches: 训练总批次数
            preprocessed_path: 预处理数据路径
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labeled_ratio = labeled_ratio
        self.total_batches = total_batches
        self.preprocessed_path = preprocessed_path or os.path.join(data_path, "augmented")
                    
        # 将有标签数据分成k组，用于均匀分布在训练批次中
        self.batch_groups = self._create_batch_groups()
        self.current_group_idx = 0
        
        # 当前批次计数
        self.current_batch = 0
        

    
    def _split_data(self):
        """划分有标签和无标签数据"""
        np.random.shuffle(self.data)
        num_labeled = int(len(self.data) * self.labeled_ratio)
        
        self.labeled_data = self.data[:num_labeled]
        self.unlabeled_data = self.data[num_labeled:]
        
        logger.info(f"有标签数据数量: {len(self.labeled_data)}")
        logger.info(f"无标签数据数量: {len(self.unlabeled_data)}")
        
    def _create_batch_groups(self):
        """
        将有标签数据分成k组,以便在训练过程中均匀分布
        k的值基于总批次数和有标签数据数量
        """
        labeled_data = self.labeled_data.copy()
        random.shuffle(labeled_data)
        
        # 确定分组数量，至少为1
        k = min(len(labeled_data), self.total_batches)
        k = max(1, k)
        
        # 将有标签数据均匀分成k组
        groups = []
        items_per_group = len(labeled_data) // k
        remainder = len(labeled_data) % k
        
        start = 0
        for i in range(k):
            group_size = items_per_group + (1 if i < remainder else 0)
            end = start + group_size
            groups.append(labeled_data[start:end])
            start = end
            
        return groups
    
    def get_next_batch(self, batch_size):
        """
        获取下一个批次的数据，保证有标签数据均匀分布
        
        参数:
            batch_size: 批次大小
            
        返回:
            批次数据字典
        """
        # 更新当前批次计数
        self.current_batch += 1
        
        # 获取当前批次的有标签数据组
        group_idx = (self.current_batch - 1) % len(self.batch_groups)
        labeled_batch = self.batch_groups[group_idx]
        
        # 从无标签数据中随机选择
        unlabeled_batch_size = batch_size - len(labeled_batch)
        if len(self.unlabeled_data) > 0:
            unlabeled_batch = random.sample(self.unlabeled_data, min(unlabeled_batch_size, len(self.unlabeled_data)))
        else:
            unlabeled_batch = []
        
        # 合并有标签和无标签数据
        batch = []
        
        # 处理有标签数据
        for item in labeled_batch:
            # 获取数据项
            if isinstance(item, dict):
                audio_features = item.get("audio_features")
                graph_features = item.get("graph_features")
                speaker_ids = item.get("speaker_ids")
                label = item.get("label")
            else:
                # 假设是元组
                audio_features, graph_features, speaker_ids, label = item
            
            batch.append({
                "audio_features": audio_features,
                "weak_audio_features": audio_features,  # 有标签数据不需要弱增强
                "strong_audio_features": audio_features,  # 有标签数据不需要强增强
                "graph_features": graph_features,
                "speaker_ids": speaker_ids,
                "label": label,
                "is_labeled": True
            })
        
        # 处理无标签数据
        for item in unlabeled_batch:
                batch.append({
                    "audio_features": item.get("original_features", item.get("audio_features")),
                    "weak_audio_features": item.get("weak_audio_features"),
                    "strong_audio_features": item.get("strong_audio_features"),
                    "graph_features": item.get("graph_features"),
                    "speaker_ids": item.get("speaker_ids"),
                    "label": item.get("label", 0),  # 默认标签
                    "is_labeled": False
                })
                
                # 对无标签数据应用弱增强和强增强
                weak_audio_features = self.weak_augment(audio_features)
                strong_audio_features = self.strong_augment(audio_features)
                
                batch.append({
                    "audio_features": audio_features,  # 原始特征
                    "weak_audio_features": weak_audio_features,  # 弱增强特征
                    "strong_audio_features": strong_audio_features,  # 强增强特征
                    "graph_features": graph_features,
                    "speaker_ids": speaker_ids,
                    "label": label,
                    "is_labeled": False
                })
        
        return self.collate_fn(batch)
    
    def collate_fn(self, batch):
        """
        整理批次数据
        
        参数:
            batch: 批次数据
            
        返回:
            整理好的批次数据
        """
        audio_features = torch.stack([item["audio_features"] for item in batch])
        weak_audio_features = torch.stack([item["weak_audio_features"] for item in batch])
        strong_audio_features = torch.stack([item["strong_audio_features"] for item in batch])
        graph_features = torch.stack([item["graph_features"] for item in batch])
        speaker_ids = torch.stack([item["speaker_ids"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        is_labeled = torch.tensor([item["is_labeled"] for item in batch])
        
        # 构建标准提示 - 使用原始特征和弱增强特征
        orig_prompts = [self.construct_prompt(audio, graph) for audio, graph in zip(audio_features, graph_features)]
        weak_prompts = [self.construct_prompt(audio, graph) for audio, graph in zip(weak_audio_features, graph_features)]
        strong_prompts = [self.construct_prompt(audio, graph) for audio, graph in zip(strong_audio_features, graph_features)]
        
        # 构建置信度提示 - 用于获取置信度分布
        confidence_prompts = [self.construct_confidence_prompt(audio, graph) for audio, graph in zip(weak_audio_features, graph_features)]
        
        # 编码标准提示
        orig_encodings = self.tokenizer(
            orig_prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        weak_encodings = self.tokenizer(
            weak_prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        strong_encodings = self.tokenizer(
            strong_prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 编码置信度提示
        confidence_encodings = self.tokenizer(
            confidence_prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 添加图特征的原始形式，用于后续直接融合
        return {
            "input_ids": orig_encodings.input_ids,
            "attention_mask": orig_encodings.attention_mask,
            "weak_input_ids": weak_encodings.input_ids,
            "weak_attention_mask": weak_encodings.attention_mask,
            "strong_input_ids": strong_encodings.input_ids,
            "strong_attention_mask": strong_encodings.attention_mask,
            "confidence_input_ids": confidence_encodings.input_ids,
            "confidence_attention_mask": confidence_encodings.attention_mask,
            "audio_features": audio_features,
            "weak_audio_features": weak_audio_features,
            "strong_audio_features": strong_audio_features,
            "graph_features": graph_features,
            "speaker_ids": speaker_ids,
            "labels": labels,
            "is_labeled": is_labeled,
            # 添加原始提示文本，方便调试
            "orig_prompts": orig_prompts,
            "weak_prompts": weak_prompts,
            "strong_prompts": strong_prompts,
            "confidence_prompts": confidence_prompts
        }
    
    def construct_prompt(self, audio_features, graph_features):
        """
        构建融合图结构信息的提示
        
        参数:
            audio_features: 音频特征
            graph_features: 图特征
            
        返回:
            构建好的提示文本
        """
        # 将特征转换为文本形式
        audio_str = self._features_to_str(audio_features)
        
        # 构建更详细的图结构描述
        if isinstance(graph_features, torch.Tensor) and graph_features.ndim > 1:
            # 提取图的结构信息
            num_nodes = graph_features.size(0)
            graph_summary = f"对话包含{num_nodes}个交互节点，"
            graph_summary += f"特征维度为{graph_features.size(1)}，"
            graph_summary += f"关键节点特征：{self._features_to_str(graph_features[0])}"
        else:
            graph_summary = self._features_to_str(graph_features)
            
        # 构建融合图信息的提示
        prompt = (
            f"请分析以下对话数据:\n"
            f"图结构信息: {graph_summary}\n"
            f"音频特征: {audio_str}\n"
            f"根据上述信息，分析用户的购买意向。"
        )
        
        return prompt
    
    def construct_confidence_prompt(self, audio_features, graph_features):
        """
        构建用于获取置信度分布的提示，融合图结构信息
        
        参数:
            audio_features: 音频特征
            graph_features: 图特征
            
        返回:
            用于获取置信度分布的提示文本
        """
        # 构建基本提示
        base_prompt = self.construct_prompt(audio_features, graph_features)
        
        # 添加置信度输出格式要求
        prompt = (
            f"{base_prompt}\n\n"
            f"请按照以下格式输出分析结果:\n"
            f"分析: [您的分析内容]\n"
            f"类别: [用户购买意向类别，如'高意向'、'中意向'、'低意向'或'无意向']\n"
            f"置信度分布: [四个类别的概率分布，格式为'高意向:0.7,中意向:0.2,低意向:0.05,无意向:0.05']\n"
        )
        
        return prompt
    
    def _features_to_str(self, features):
        """将特征转换为更可读的文本形式"""
        # 优化特征文本表示，避免过长数字列表
        if isinstance(features, torch.Tensor) and features.ndim > 1:
            # 对于多维特征，提取关键统计信息
            stats = {
                "均值": f"{features.mean().item():.3f}",
                "最大值": f"{features.max().item():.3f}",
                "最小值": f"{features.min().item():.3f}"
            }
            return f"[特征统计: {stats}]"
        elif isinstance(features, torch.Tensor):
            # 如果是1D张量，只取前几个值和基本统计量
            if (len(features) > 5):
                return f"[特征样本: {features[:5].tolist()}, 均值: {features.mean().item():.3f}]"
            else:
                return str(features.tolist())
        else:
            return str(features)

    def _load_preprocessed_data(self):
        """加载预处理好的数据"""
        logger.info(f"加载预处理数据: {self.preprocessed_path}")
        
        try:
            # 检查是否存在配对数据文件
            paired_path = os.path.join(self.preprocessed_path, "fixmatch_paired_data.pt")
            
            if os.path.exists(paired_path):
                logger.info(f"加载预处理的FixMatch配对数据: {paired_path}")
                self.unlabeled_paired_data = torch.load(paired_path)
                logger.info(f"成功加载无标签配对数据: {len(self.unlabeled_paired_data)}个样本")
                
                # 有标签数据仍然需要从原始数据中加载
                labeled_data_path = os.path.join(self.data_path, "labeled_data.pt")
                if os.path.exists(labeled_data_path):
                    self.labeled_data = torch.load(labeled_data_path)
                    logger.info(f"成功加载有标签数据: {len(self.labeled_data)}个样本")
                else:
                    # 如果找不到独立的有标签数据文件，从原始数据中提取
                    self.data = self._load_data()
                    # 按比例提取有标签数据
                    num_labeled = int(len(self.data) * self.labeled_ratio)
                    self.labeled_data = self.data[:num_labeled]
                    logger.info(f"从原始数据中提取有标签数据: {len(self.labeled_data)}个样本")
                
                # 将无标签配对数据设置为无标签数据
                self.unlabeled_data = self.unlabeled_paired_data
                
                # 原始数据是有标签和无标签数据的合并
                self.data = self.labeled_data + self.unlabeled_data
                
                return
            
            # 如果没有配对数据，尝试分别加载弱增强和强增强数据
            weak_path = os.path.join(self.preprocessed_path, "weak_augmented_data.pt")
            strong_path = os.path.join(self.preprocessed_path, "strong_augmented_data.pt")
            
            if os.path.exists(weak_path) and os.path.exists(strong_path):
                logger.info(f"加载单独的弱增强数据: {weak_path}")
                logger.info(f"加载单独的强增强数据: {strong_path}")
                
                weak_data = torch.load(weak_path)
                strong_data = torch.load(strong_path)
                logger.info(f"成功加载弱增强数据: {len(weak_data)}个样本")
                logger.info(f"成功加载强增强数据: {len(strong_data)}个样本")
                
                # 合并弱增强和强增强数据为配对数据
                # 假设两者样本数量相同且顺序对应
                self.unlabeled_paired_data = []
                if len(weak_data) == len(strong_data):
                    orig_samples = [s for s in weak_data if s.get("augmentation") == "original"]
                    weak_samples = [s for s in weak_data if s.get("augmentation") == "weak"]
                    strong_samples = strong_data
                    
                    # 确保样本数量匹配
                    min_len = min(len(orig_samples), len(weak_samples), len(strong_samples))
                    if min_len > 0:
                        for i in range(min_len):
                            paired_sample = {k: v for k, v in orig_samples[i].items() if k != "augmentation"}
                            paired_sample["original_features"] = orig_samples[i].get("audio_features").clone()
                            paired_sample["weak_audio_features"] = weak_samples[i].get("audio_features").clone()
                            paired_sample["strong_audio_features"] = strong_samples[i].get("audio_features").clone()
                            paired_sample["is_labeled"] = False
                            self.unlabeled_paired_data.append(paired_sample)
                        
                        logger.info(f"成功合并为配对数据: {len(self.unlabeled_paired_data)}个样本")
                    else:
                        logger.warning("无法配对弱增强和强增强数据，回退到标准处理")
                        self.data = self._load_data()
                        self._split_data()
                        return
                else:
                    logger.warning("弱增强和强增强数据数量不匹配，回退到标准处理")
                    self.data = self._load_data()
                    self._split_data()
                    return
                
                # 有标签数据仍然需要从原始数据中加载
                labeled_data_path = os.path.join(self.data_path, "labeled_data.pt")
                if os.path.exists(labeled_data_path):
                    self.labeled_data = torch.load(labeled_data_path)
                    logger.info(f"成功加载有标签数据: {len(self.labeled_data)}个样本")
                else:
                    # 如果找不到独立的有标签数据文件，从原始数据中提取
                    orig_data = self._load_data()
                    # 按比例提取有标签数据
                    num_labeled = int(len(orig_data) * self.labeled_ratio)
                    self.labeled_data = orig_data[:num_labeled]
                    logger.info(f"从原始数据中提取有标签数据: {len(self.labeled_data)}个样本")
                
                # 将无标签配对数据设置为无标签数据
                self.unlabeled_data = self.unlabeled_paired_data
                
                # 原始数据是有标签和无标签数据的合并
                self.data = self.labeled_data + self.unlabeled_data
                
                return
            
            # 如果找不到预处理数据，回退到标准处理
            logger.warning(f"未找到预处理数据，回退到标准处理")
            self.data = self._load_data()
            self._split_data()
        
        except Exception as e:
            logger.error(f"加载预处理数据失败: {e}")
            logger.warning("回退到标准处理")
            self.data = self._load_data()
            self._split_data()


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
        preprocessed_path=None,
        max_length=512,
        system_prompt=None,
        num_segments=None
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
            preprocessed_path: 预处理数据路径
            max_length: 最大序列长度
            system_prompt: 系统提示文本
            num_segments: 每个音频切分的片段数量，None表示随机
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
        
        # 创建数据加载器
        train_loader = create_telemarketing_dataloader(
            data_path=train_data_path,
            encoder=AudioEncoder.create_default_encoder(),  # 添加音频编码器
            labels_file='migrated_labels.csv',              # 使用migrated_labels.csv
            max_segments=num_segments or 10,                # 使用num_segments或默认值
            cache_dir='features_cache',                     # 使用特征缓存目录
            batch_size=8,                                   # 设置批次大小
            shuffle=True,                                   # 打乱数据顺序
            system_prompt=system_prompt,                    # 系统提示文本
            num_workers=0,                                  # 避免多进程问题
            balance_labels=True                             # 启用标签均衡采样
        )
        
        if val_data_path:
            val_loader = create_telemarketing_dataloader(
                data_path=val_data_path,
                encoder=AudioEncoder.create_default_encoder(),  # 添加音频编码器
                labels_file='migrated_labels.csv',              # 使用migrated_labels.csv
                max_segments=num_segments or 10,                # 使用num_segments或默认值
                cache_dir='features_cache',                     # 使用特征缓存目录
                batch_size=8,                                   # 设置批次大小
                shuffle=False,                                  # 验证集不打乱顺序
                system_prompt=system_prompt,                    # 系统提示文本
                num_workers=0,                                  # 避免多进程问题
                balance_labels=False                            # 验证集不需要标签均衡采样
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
                phone_ids = batch['phone_ids']
                original_features = [feat.to(self.device) for feat in batch['original_features']]
                segment_features_list = batch['segment_features']
                labels = batch['labels'].to(self.device)
                label_onehots = batch['label_onehots'].to(self.device)
                num_segments = batch['num_segments']
                
                # 处理图特征和说话人ID - 使用新格式
                graph_embeddings = None
                if "speakers" in batch:
                    speaker_ids = batch['speakers']
                    # 这里可以添加图模型处理，如果需要
                
                # 构建适合模型的输入格式
                # 这里将原始音频特征和片段特征转换为模型期望的格式
                
                # 处理原始音频特征
                try:
                    # 构造模型输入 - 可能需要根据模型期望格式调整
                    original_inputs = {
                        "input_ids": torch.ones(1, 4).long().to(self.device),  # 占位符
                        "attention_mask": torch.ones(1, 4).to(self.device),    # 占位符
                        "labels": labels                                      # 使用dataloader提供的标签
                    }
                    
                    # 如果你的模型需要直接使用特征，添加到inputs中
                    original_inputs["features"] = original_features[0] if len(original_features) > 0 else None
                    
                    original_outputs = self.model(**original_inputs)
                    orig_loss = original_outputs.loss
                    original_loss += orig_loss.item()
                except Exception as e:
                    logger.error(f"处理原始音频时出错: {e}")
                    orig_loss = torch.tensor(0.0, device=self.device)
                
                # 2. 处理音频片段
                segments_batch_loss = 0
                num_valid_segments = 0
                
                # 遍历每个样本的所有片段
                for batch_idx, segments in enumerate(segment_features_list):
                    for segment_idx, segment_feature in enumerate(segments):
                        try:
                            # 将特征移动到设备
                            segment_feature = segment_feature.to(self.device)
                            
                            # 构造片段输入
                            segment_inputs = {
                                "input_ids": torch.ones(1, 4).long().to(self.device),  # 占位符
                                "attention_mask": torch.ones(1, 4).to(self.device),    # 占位符
                                "labels": labels[batch_idx:batch_idx+1],               # 使用对应样本的标签
                                "features": segment_feature                           # 片段特征
                            }
                            
                            # 前向传播
                            segment_outputs = self.model(**segment_inputs)
                            segments_batch_loss += segment_outputs.loss
                            num_valid_segments += 1
                        except Exception as e:
                            logger.error(f"处理音频片段时出错: {e}")
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
                original_features = [feat.to(self.device) for feat in batch['original_features']]
                labels = batch['labels'].to(self.device)
                label_onehots = batch['label_onehots'].to(self.device)
                
                # 前向传播
                try:
                    # 构造模型输入
                    inputs = {
                        "input_ids": torch.ones(len(labels), 4).long().to(self.device),  # 占位符
                        "attention_mask": torch.ones(len(labels), 4).to(self.device),    # 占位符
                        "labels": labels
                    }
                    
                    # 如果模型需要直接使用特征，添加到inputs中
                    for i, feat in enumerate(original_features):
                        if i == 0:  # 只处理第一个样本，用于示例
                            inputs["features"] = feat
                    
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    # 如果模型输出包含logits，则获取预测结果
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    logger.error(f"评估时处理批次出错: {e}")
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
    from qwen2_5_omni_light import Qwen25OmniLightProcessor, Qwen2_5OmniTextOnlyModel
    import torch
    from qwen_omni_utils import process_mm_info
    
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
    
    # 更新图模型配置
    graph_config = {
        "token_embedding_dim": 768,  # 输入token嵌入维度
        "hidden_dim": 256,          # 隐藏层维度
        "output_dim": 128,          # 输出维度
        "num_heads": 4,             # 注意力头数量
        "speaker_embedding_dim": 64, # 说话者嵌入维度
        "num_speakers": None,       # 动态说话者嵌入
        "num_layers": 2,            # GAT层数
        "dropout": 0.2,             # Dropout概率
        "similarity_threshold": 0.5,# 相似度阈值
        "context_window_size": 4,   # 上下文窗口大小
        "aggregation_method": "mean"# 特征聚合方法
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
