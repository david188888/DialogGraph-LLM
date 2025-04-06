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

from utils.graph import DialogueGraphModel
from utils.audio_augment import AudioAugmenter

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
            if len(features) > 5:
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
    """半监督自适应阈值微调训练器, 整合FixMatch算法和类别自适应阈值策略"""
    
    def __init__(
        self, 
        model_name, 
        graph_config, 
        output_dir, 
        device="cuda",
        learning_rate=5e-5,
        weight_decay=0.01,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        threshold_init=0.7,
        threshold_momentum=0.7,
        confidence_weight=0.5,
        ema_decay=0.999,  # 指数移动平均衰减率
        num_classes=4,    # 分类任务的类别数量
        class_threshold_momentum=0.9  # 类别阈值的EMA动量
    ):
        """
        初始化训练器
        
        参数:
            model_name: 预训练模型名称
            graph_config: 图模型配置
            output_dir: 输出目录
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            lora_r: LoRA秩
            lora_alpha: LoRA缩放因子
            lora_dropout: LoRA dropout概率
            threshold_init: 阈值初始值
            threshold_momentum: 阈值动量
            confidence_weight: 置信度权重
            ema_decay: 指数移动平均衰减率
            num_classes: 分类任务的类别数量
            class_threshold_momentum: 类别阈值的EMA动量
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # LoRA配置
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query_key_value", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"], # 这个是关于Lora模块
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 全局自适应阈值参数
        self.threshold = threshold_init
        self.threshold_momentum = threshold_momentum
        self.confidence_weight = confidence_weight
        
        # 指数移动平均衰减率
        self.ema_decay = ema_decay
        
        # 类别自适应阈值参数
        self.num_classes = num_classes
        self.class_threshold_momentum = class_threshold_momentum
        # 初始化每个类别的阈值
        self.class_thresholds = torch.ones(num_classes, device=device) * threshold_init
        # 类别概率的EMA
        self.class_probs_ema = torch.ones(num_classes, device=device) / num_classes
        
        # 初始化模型和分词器
        self._init_model_and_tokenizer()
        
        # 初始化图模型
        self.graph_model = DialogueGraphModel(**graph_config).to(device)
        
        # 初始化优化器
        self._init_optimizer()
        
        # 动态损失权重
        self.labeled_weight = 1.0
        self.unlabeled_weight = 0.0  # 初始时不使用无标签数据
        
    def _init_model_and_tokenizer(self):
        """初始化模型和分词器"""
        logger.info(f"加载模型: {self.model_name}")
        

        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device
        )
        
        # 准备模型进行低精度训练
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 应用LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        
    def _init_optimizer(self):
        """初始化优化器"""
        # 收集需要优化的参数
        optimizer_grouped_parameters = [
            # LoRA参数
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
    
    def _parse_confidence_output(self, output_text):
        """
        解析LLM输出的置信度分布文本
        
        参数:
            output_text: LLM生成的文本
            
        返回:
            解析后的置信度分布，形状为 [num_classes]
        """
        confidence_pattern = r"置信度分布:\s*(高意向:[\d.]+,中意向:[\d.]+,低意向:[\d.]+,无意向:[\d.]+)"
        class_pattern = r"(高意向|中意向|低意向|无意向):([\d.]+)"
        
        # 默认的置信度分布（均匀分布）
        default_confidence = torch.ones(self.num_classes, device=self.device) / self.num_classes
        
        try:
            # 尝试匹配置信度分布部分
            import re
            confidence_match = re.search(confidence_pattern, output_text)
            
            if confidence_match:
                confidence_text = confidence_match.group(1)
                # 找到所有类别和对应的概率
                class_matches = re.findall(class_pattern, confidence_text)
                
                if len(class_matches) == self.num_classes:
                    # 用于存储解析后的置信度
                    confidence = torch.zeros(self.num_classes, device=self.device)
                    
                    # 类别到索引的映射，这个需要后续调整
                    class_to_idx = {
                        "高意向": 0,
                        "中意向": 1,
                        "低意向": 2,
                        "无意向": 3
                    }
                    
                    # 解析每个类别的概率
                    for class_name, prob in class_matches:
                        idx = class_to_idx.get(class_name, -1)
                        if idx >= 0:
                            confidence[idx] = float(prob)
                    
                    # 归一化
                    if confidence.sum() > 0:
                        confidence = confidence / confidence.sum()
                        return confidence
            
            # 如果解析失败，返回默认分布
            return default_confidence
            
        except Exception as e:
            logger.warning(f"解析置信度分布失败: {e}")
            return default_confidence
    
    def _get_confidence_distribution(self, model_outputs, input_ids):
        """
        从模型输出中获取置信度分布
        
        参数:
            model_outputs: 模型输出
            input_ids: 输入ID
            
        返回:
            置信度分布，形状为 [batch_size, num_classes]
        """
        batch_size = input_ids.size(0)
        confidences = torch.zeros(batch_size, self.num_classes, device=self.device)
        
        # 生成完整的文本输出
        output_ids = model_outputs.logits.argmax(dim=-1)
        
        for i in range(batch_size):
            # 获取生成的文本
            generated_ids = output_ids[i]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 解析置信度分布
            confidence = self._parse_confidence_output(generated_text)
            confidences[i] = confidence
        
        return confidences
    
    def _update_thresholds(self, confidence_probs):
        """
        更新自适应阈值
        
        参数:
            confidence_probs: 置信度概率分布 [batch_size, num_classes]
            
        返回:
            更新后的每个类别的自适应阈值
        """
        batch_size = confidence_probs.size(0)
        
        # 1. 计算每个样本的最大预测概率
        max_probs, pred_classes = torch.max(confidence_probs, dim=1)
        
        # 2. 更新全局阈值 τ_s
        # 使用EMA更新全局阈值
        mean_confidence = max_probs.mean().item()
        self.threshold = self.threshold_momentum * self.threshold + (1 - self.threshold_momentum) * mean_confidence
        
        # 3. 更新每个类别的局部阈值 p̃_s(k)
        # 计算每个类别在当前批次中的平均概率 Q_a,b(k)
        class_batch_probs = torch.zeros(self.num_classes, device=self.device)
        
        # 对每个类别，计算批次中所有样本在该类别上的平均概率
        for k in range(self.num_classes):
            # 累加该类别的所有概率
            class_batch_probs[k] = confidence_probs[:, k].mean()
        
        # 使用EMA更新类别局部阈值
        # p̃_s(k) = λp̃_s-1(k) + (1-λ)(1/B_a) * ∑ Q_a,b(k)
        self.class_probs_ema = (
            self.class_threshold_momentum * self.class_probs_ema + 
            (1 - self.class_threshold_momentum) * class_batch_probs
        )
        
        # 4. 根据最大值归一化，生成最终自适应阈值
        # 找到最大的类别局部阈值
        max_class_prob = torch.max(self.class_probs_ema)
        
        # 根据公式: τ_cs(k) = (p̃_s(k) / max_k' p̃_s(k')) * τ_s
        if max_class_prob > 0:
            # 每个类别阈值 = (类别局部阈值/最大局部阈值) * 全局阈值
            class_thresholds = (self.class_probs_ema / max_class_prob) * self.threshold
        else:
            # 如果所有类别概率为0，则使用全局阈值
            class_thresholds = torch.ones(self.num_classes, device=self.device) * self.threshold
        
        return class_thresholds
    
    def _generate_pseudo_labels(self, confidence_probs, class_thresholds):
        """
        根据置信度概率和自适应阈值生成伪标签
        
        参数:
            confidence_probs: 置信度概率分布 [batch_size, num_classes]
            class_thresholds: 每个类别的自适应阈值 [num_classes]
            
        返回:
            伪标签和置信度掩码
        """
        batch_size = confidence_probs.size(0)
        
        # 获取最大概率及其对应的类别
        max_probs, pseudo_labels = torch.max(confidence_probs, dim=1)
        
        # 计算置信度掩码
        # 对于每个样本，检查其最大概率是否超过对应类别的自适应阈值
        thresholds = class_thresholds[pseudo_labels]
        confidence_mask = (max_probs > thresholds)
        
        return pseudo_labels, confidence_mask
    
    def train(
        self, 
        train_dataset, 
        val_dataset=None, 
        batch_size=8, 
        num_epochs=5, 
        gradient_accumulation_steps=4,
        log_interval=10,
        eval_interval=100,
        save_interval=1000,
        use_wandb=False,
        lambda_u_max=1.0,  # 无标签损失的最大权重
        lambda_u_rampup_epochs=1.0  # 无标签损失权重增加的轮数
       ):
        """
        训练模型，实现FixMatch算法的半监督学习
        
        参数:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            batch_size: 批次大小
            num_epochs: 训练轮数
            gradient_accumulation_steps: 梯度累积步数
            log_interval: 日志记录间隔
            eval_interval: 评估间隔
            save_interval: 保存间隔
            use_wandb: 是否使用wandb记录实验
            lambda_u_max: 无标签损失的最大权重
            lambda_u_rampup_epochs: 无标签损失权重增加的轮数
        """
        # 初始化wandb
 
        if use_wandb:
            wandb.init(
                project="audio-llm-telemarketing",
                config={
                    "model_name": self.model_name,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "lora_r": self.lora_config.r,
                    "lora_alpha": self.lora_config.lora_alpha,
                    "lora_dropout": self.lora_config.lora_dropout,
                    "threshold_init": self.threshold,
                    "threshold_momentum": self.threshold_momentum,
                    "confidence_weight": self.confidence_weight,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "lambda_u_max": lambda_u_max,
                    "lambda_u_rampup_epochs": lambda_u_rampup_epochs,
                    "algorithm": "FixMatch"
                }
            )
        
        # 使用自定义批次获取方法，确保有标签数据均匀分布
        train_dataset.total_batches = num_epochs * (len(train_dataset) // batch_size)
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=val_dataset.collate_fn
            )
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 训练循环
        global_step = 0
        total_steps = train_dataset.total_batches
        
        for epoch in range(num_epochs):
            self.model.train()
            self.graph_model.train()
            
            epoch_loss = 0
            labeled_loss = 0
            unlabeled_loss = 0
            num_labeled = 0
            num_unlabeled = 0
            num_masked = 0  # 记录被掩码（置信度低）的样本数
            
            # 计算无标签损失的权重（lambda_u）- 逐渐增加
            lambda_u = lambda_u_max * min(1.0, epoch / lambda_u_rampup_epochs)
            
            num_batches = len(train_dataset) // batch_size
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step in progress_bar:
                # 获取当前批次数据
                batch = train_dataset.get_next_batch(batch_size)
                
                # 将批次数据移动到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                weak_input_ids = batch["weak_input_ids"].to(self.device)
                weak_attention_mask = batch["weak_attention_mask"].to(self.device)
                strong_input_ids = batch["strong_input_ids"].to(self.device)
                strong_attention_mask = batch["strong_attention_mask"].to(self.device)
                audio_features = batch["audio_features"].to(self.device)
                weak_audio_features = batch["weak_audio_features"].to(self.device)
                strong_audio_features = batch["strong_audio_features"].to(self.device)
                graph_features = batch["graph_features"].to(self.device)
                speaker_ids = batch["speaker_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                is_labeled = batch["is_labeled"].to(self.device)
                confidence_input_ids = batch["confidence_input_ids"].to(self.device)
                confidence_attention_mask = batch["confidence_attention_mask"].to(self.device)
                
                # 使用图模型处理图特征
                graph_embeddings = self.graph_model(graph_features, speaker_ids)
                
                # 计算有标签数据的损失
                labeled_mask = is_labeled.bool()
                
                if labeled_mask.any():
                    labeled_outputs = self.model(
                        input_ids=input_ids[labeled_mask],
                        attention_mask=attention_mask[labeled_mask],
                        labels=input_ids[labeled_mask]  # 使用输入作为标签进行自回归训练
                    )
                    
                    current_labeled_loss = labeled_outputs.loss
                    labeled_loss += current_labeled_loss.item()
                    num_labeled += labeled_mask.sum().item()
                else:
                    current_labeled_loss = 0
                
                # 计算无标签数据的损失（使用FixMatch方法和自适应阈值）
                unlabeled_mask = ~labeled_mask
                
                if unlabeled_mask.any() and lambda_u > 0:
                    # 1. 使用置信度提示获取每个样本的置信度分布
                    with torch.no_grad():
                        confidence_outputs = self.model(
                            input_ids=confidence_input_ids[unlabeled_mask],
                            attention_mask=confidence_attention_mask[unlabeled_mask],
                            output_hidden_states=False
                        )
                        
                        # 解析模型输出，获取置信度分布
                        confidence_probs = self._get_confidence_distribution(
                            confidence_outputs, 
                            confidence_input_ids[unlabeled_mask]
                        )
                        
                        # 更新自适应阈值
                        class_thresholds = self._update_thresholds(confidence_probs)
                        
                        # 生成伪标签和置信度掩码
                        pseudo_labels, confidence_mask = self._generate_pseudo_labels(
                            confidence_probs, 
                            class_thresholds
                        )
                    
                    # 2. 使用强增强数据和伪标签进行训练
                    strong_outputs = self.model(
                        input_ids=strong_input_ids[unlabeled_mask],
                        attention_mask=strong_attention_mask[unlabeled_mask]
                    )
                    
                    strong_logits = strong_outputs.logits
                    
                    # 3. 计算无标签损失 - 使用CrossEntropyLoss简化计算
                    if confidence_mask.sum() > 0:
                        # 处理模型输出的logits形状以适应CrossEntropyLoss
                        # 假设强增强模型输出的logits形状为[batch_size, sequence_length, vocab_size]
                        # 我们需要处理成适合分类任务的形状
                        
                        # 使用CrossEntropyLoss计算损失
                        loss_fn = nn.CrossEntropyLoss(reduction="none")
                        
                        # 适配LLM输出，选择适当的token位置计算损失
                        relevant_logits = strong_logits[:, -1, :]  # 可能需要根据实际情况调整
                        
                        # 计算每个样本的损失
                        individual_losses = loss_fn(relevant_logits, pseudo_labels)
                        
                        # 应用置信度掩码
                        masked_losses = individual_losses * confidence_mask.float()
                        
                        # 计算平均损失
                        current_unlabeled_loss = masked_losses.sum() / confidence_mask.sum()
                    else:
                        current_unlabeled_loss = torch.tensor(0.0, device=self.device)
                    
                    unlabeled_loss += current_unlabeled_loss.item()
                    num_unlabeled += unlabeled_mask.sum().item()
                    num_masked += (~confidence_mask).sum().item()
                else:
                    current_unlabeled_loss = 0
                
                # 计算总损失 - 使用动态加权
                if isinstance(current_labeled_loss, torch.Tensor) and isinstance(current_unlabeled_loss, torch.Tensor):
                    loss = current_labeled_loss + lambda_u * current_unlabeled_loss
                elif isinstance(current_labeled_loss, torch.Tensor):
                    loss = current_labeled_loss
                elif isinstance(current_unlabeled_loss, torch.Tensor):
                    loss = lambda_u * current_unlabeled_loss
                else:
                    loss = torch.tensor(0.0, device=self.device)
                
                # 累积梯度
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                # 梯度累积
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == num_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        "loss": epoch_loss / (step + 1),
                        "labeled_loss": labeled_loss / max(num_labeled, 1),
                        "unlabeled_loss": unlabeled_loss / max(num_unlabeled, 1),
                        "threshold": self.threshold,
                        "lambda_u": lambda_u,
                        "masked_ratio": num_masked / max(num_unlabeled, 1)
                    })
                    
                    # 记录日志
                    if global_step % log_interval == 0 and use_wandb:
                        # 记录每个类别的自适应阈值
                        class_threshold_dict = {
                            f"class_{i}_threshold": self.class_thresholds[i].item() 
                            for i in range(self.num_classes)
                        }
                        
                        # 记录每个类别的EMA概率
                        class_prob_dict = {
                            f"class_{i}_prob_ema": self.class_probs_ema[i].item() 
                            for i in range(self.num_classes)
                        }
                        
                        # 合并所有要记录的信息
                        log_dict = {
                            "loss": epoch_loss / (step + 1),
                            "labeled_loss": labeled_loss / max(num_labeled, 1),
                            "unlabeled_loss": unlabeled_loss / max(num_unlabeled, 1),
                            "global_threshold": self.threshold,
                            "lambda_u": lambda_u,
                            "masked_ratio": num_masked / max(num_unlabeled, 1),
                            "global_step": global_step,
                            "progress": global_step / total_steps
                        }
                        log_dict.update(class_threshold_dict)
                        log_dict.update(class_prob_dict)
                        
                        wandb.log(log_dict)
                    
                    # 评估模型
                    if val_dataset and global_step % eval_interval == 0:
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
            epoch_avg_loss = epoch_loss / num_batches
            labeled_avg_loss = labeled_loss / max(num_labeled, 1)
            unlabeled_avg_loss = unlabeled_loss / max(num_unlabeled, 1)
            masked_ratio = num_masked / max(num_unlabeled, 1)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed:")
            logger.info(f"  Average loss: {epoch_avg_loss:.4f}")
            logger.info(f"  Labeled loss: {labeled_avg_loss:.4f}")
            logger.info(f"  Unlabeled loss: {unlabeled_avg_loss:.4f}")
            logger.info(f"  Global threshold: {self.threshold:.4f}")
            logger.info(f"  Lambda_u: {lambda_u:.4f}")
            logger.info(f"  Masked ratio: {masked_ratio:.4f}")
            
            # 记录每个类别的自适应阈值
            logger.info("  类别自适应阈值:")
            for i in range(self.num_classes):
                logger.info(f"    类别 {i}: {self.class_thresholds[i].item():.4f}")
            
            if use_wandb:
                # 记录每个类别的自适应阈值
                class_threshold_dict = {
                    f"epoch_class_{i}_threshold": self.class_thresholds[i].item() 
                    for i in range(self.num_classes)
                }
                
                # 记录每个类别的EMA概率
                class_prob_dict = {
                    f"epoch_class_{i}_prob_ema": self.class_probs_ema[i].item() 
                    for i in range(self.num_classes)
                }
                
                # 合并所有要记录的信息
                log_dict = {
                    "epoch": epoch + 1,
                    "epoch_loss": epoch_avg_loss,
                    "epoch_labeled_loss": labeled_avg_loss,
                    "epoch_unlabeled_loss": unlabeled_avg_loss,
                    "epoch_global_threshold": self.threshold,
                    "epoch_lambda_u": lambda_u,
                    "epoch_masked_ratio": masked_ratio
                }
                log_dict.update(class_threshold_dict)
                log_dict.update(class_prob_dict)
                
                wandb.log(log_dict)
            
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
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # 将批次数据移动到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                audio_features = batch["audio_features"].to(self.device)
                graph_features = batch["graph_features"].to(self.device)
                speaker_ids = batch["speaker_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 使用图模型处理图特征
                graph_embeddings = self.graph_model(graph_features, speaker_ids)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # 使用输入作为标签进行自回归训练
                )
                
                val_loss += outputs.loss.item()
                
                # 获取预测结果
                # 注意：在实际项目中，这里应该根据实际任务进行预测提取
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                # 收集预测和标签
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        
        # 计算评估指标
        # 注意：在实际项目中，这里应该根据实际任务计算适当的指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )
        
        avg_loss = val_loss / len(val_loader)
        
        return {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1
        }
    
    def save_model(self, output_path):
        """
        保存模型
        
        参数:
            output_path: 输出路径
        """
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 保存LoRA模型
        self.model.save_pretrained(output_path)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_path)
        
        # 保存图模型
        torch.save(self.graph_model.state_dict(), os.path.join(output_path, "graph_model.pt"))
        
        # 保存训练配置
        with open(os.path.join(output_path, "trainer_config.json"), "w") as f:
            import json
            json.dump({
                "model_name": self.model_name,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "threshold": self.threshold,
                "threshold_momentum": self.threshold_momentum,
                "confidence_weight": self.confidence_weight
            }, f)
        
        logger.info(f"模型保存到: {output_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        参数:
            model_path: 模型路径
        """
        # 加载LoRA模型
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载图模型
        self.graph_model.load_state_dict(torch.load(os.path.join(model_path, "graph_model.pt")))
        
        logger.info(f"模型从 {model_path} 加载成功")


# 更新main函数以使用新的训练方法和预处理的增强数据
def main(use_preprocessed=True):
    """
    主函数
    
    参数:
        use_preprocessed: 是否使用预处理好的数据
    """
    # 配置参数
    model_name = "THUDM/chatglm3-6b"  # LLM路径
    graph_config = {
        "input_dim": 768,  # 根据您的特征维度调整
        "hidden_dim": 256,
        "output_dim": 128,
        "num_heads": 4,
        "speaker_embedding_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "similarity_threshold": 0.5,
        "context_window_size": 4
    }
    output_dir = "./outputs"
    
    # 数据路径
    train_data_path = "./data/train"
    val_data_path = "./data/val"
    augmented_data_path = "./data/augmented"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(augmented_data_path, exist_ok=True)
    
    # 根据use_preprocessed参数决定是否预处理数据
    if use_preprocessed:
        # 检查是否已经存在预处理好的数据
        fixmatch_data_path = os.path.join(augmented_data_path, "fixmatch_paired_data.pt")
        if not os.path.exists(fixmatch_data_path):
            logger.info("未找到预处理好的数据，开始预处理...")
            
            # 导入预处理脚本
            try:
                import sys
                sys.path.append(".")
                from preprocess_data import preprocess_unlabeled_data
                
                # 执行预处理
                unlabeled_data_path = os.path.join(train_data_path, "unlabeled_data.pt")
                success = preprocess_unlabeled_data(
                    unlabeled_data_path,
                    augmented_data_path,
                    weak_augment=True,
                    strong_augment=True,
                    num_weak_augmentations=1,
                    num_strong_augmentations=1,
                    seed=42
                )
                
                if success:
                    logger.info("数据预处理完成，将使用预处理好的数据进行训练")
                else:
                    logger.warning("数据预处理失败，回退到动态生成数据进行训练")
                    use_preprocessed = False
            except ImportError:
                logger.warning("找不到预处理脚本，回退到动态生成数据进行训练")
                use_preprocessed = False
            except Exception as e:
                logger.warning(f"预处理数据时出错: {e}")
                logger.warning("回退到动态生成数据进行训练")
                use_preprocessed = False
        else:
            logger.info(f"发现预处理好的数据: {fixmatch_data_path}")
    
    # 初始化训练器
    trainer = AdaptiveThresholdTrainer(
        model_name=model_name,
        graph_config=graph_config,
        output_dir=output_dir,
        learning_rate=2e-5,
        lora_r=16,
        lora_alpha=32,
        threshold_init=0.7,
        num_classes=4,
        class_threshold_momentum=0.9
    )
    
    # 初始化分词器
    tokenizer = trainer.tokenizer
    
    # 加载数据集
    train_dataset = AudioDialogueDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        labeled_ratio=0.1,  # 10%有标签数据
        use_preprocessed=use_preprocessed,
        preprocessed_path=augmented_data_path
    )
    
    val_dataset = AudioDialogueDataset(
        data_path=val_data_path,
        tokenizer=tokenizer,
        labeled_ratio=1.0,  # 验证集全部有标签
        use_preprocessed=False  # 验证集通常不需要增强
    )
    
    # 训练模型
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_epochs=3,
        gradient_accumulation_steps=4,
        lambda_u_max=1.0,  # 无标签损失的最大权重
        lambda_u_rampup_epochs=1.0,  # 无标签损失权重增加的轮数
        use_wandb=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--use-preprocessed', action='store_true', default=True,
                        help='是否使用预处理好的数据')
    
    args = parser.parse_args()
    
    main(use_preprocessed=args.use_preprocessed)
