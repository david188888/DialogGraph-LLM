import sys
import os
import argparse

# 设置内存优化环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
# 添加学习率调度器相关导入
from transformers import get_scheduler, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from transformers import (
#     Qwen2_5OmniThinkerConfig
# )
from transformers import AutoConfig
import numpy as np # 添加 numpy


from ECAI.qwen2_5_omni_light import Qwen2_5OmniLightProcessor, Qwen2_5OmniTextOnlyModel


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
        device="cuda",
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
        update_dataset_freq=1,
        margin_epsilon=0.05,     # 伪标签生成的最小可接受超阈差值
        use_accelerator=True,   # 是否使用Accelerator
        # --- 添加学习率调度器相关参数 ---
        lr_scheduler_type="cosine",  # 学习率调度器类型
        warmup_ratio=0.1,                       # warmup阶段占总步数的比例
        num_training_steps=None,                # 总训练步数，如果未指定，将在train方法中计算
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
            margin_epsilon: 伪标签生成的最小可接受超阈差值
            use_accelerator: 是否使用Accelerator
            lr_scheduler_type: 学习率调度器类型，可选"linear"、"cosine"、"cosine_with_warmup"、"constant_with_warmup"
            warmup_ratio: warmup阶段占总步数的比例
            num_training_steps: 总训练步数，如果未指定，将在train方法中计算
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
        self.margin_epsilon = margin_epsilon
        self.use_accelerator = use_accelerator
        
        # --- 学习率调度器相关参数 ---
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio
        self.num_training_steps = num_training_steps
        self.lr_scheduler = None  # 将在train方法中初始化
        
        # 初始化自适应阈值参数 (仅在半监督模式下)
        if self.use_semi_supervised:
            # 确定目标设备 (优先使用传入的 self.device，通常为 'cuda' 或 'cuda:0')
            target_device = self.device 
            # 如果模型有明确的设备映射（例如多GPU），取第一个GPU
            if hasattr(self.model, 'hf_device_map'):
                # 尝试获取模型第一个参数所在的设备
                try:
                    target_device = next(self.model.parameters()).device
                except StopIteration:
                    # 如果模型没有参数，回退到 self.device
                    target_device = self.device
            elif hasattr(self.model, 'device'):
                 target_device = self.model.device
                 
            logger.info(f"Initializing semi-supervised thresholds on device: {target_device}")
            # tau: 全局置信度阈值，用于判断无标签数据的预测是否可靠，控制伪标签生成的严格程度
            self.tau = torch.tensor(self.initial_tau, device=target_device)
            # p_tilde: 类别分布估计值，记录每个类别的平均预测概率，用于处理类别不平衡问题
            self.p_tilde = torch.ones(self.num_classes, device=target_device) / self.num_classes
            # tau_c: 类别特定的置信度阈值，为每个类别定制阈值，考虑了类别不平衡情况
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
                self.model.print_trainable_parameters()
                logger.info("LoRA配置应用成功！")
            except Exception as e:
                logger.error(f"应用LoRA配置时出错: {e}")
                import traceback
                traceback.print_exc()
                logger.warning("将使用全参数微调...")
                self.use_lora = False

        # 初始化图模型
        self.graph_model = DialogueGraphModel(**graph_config).to(device)

        # 打印 graph_model 每个命名模块的设备
        print("Graph model layer devices:")
        for name, module in self.graph_model.named_modules():
            try:
                # 尝试从模块的第一个参数获取设备
                module_device = next(module.parameters()).device
                print(f" - {name}: {module_device}")
            except StopIteration:
                # 如果没有参数，尝试从缓冲区获取
                try:
                    buffer_device = next(module.buffers()).device
                    print(f" - {name}: {buffer_device} (from buffer)")
                except StopIteration:
                    # 如果也没有缓冲区，则无法确定
                    print(f" - {name}: (No parameters or buffers to determine device)")
            except AttributeError:
                # 处理其他可能的错误
                print(f" - {name}: (Could not determine device)")

        # 检查图模型是否需要转换为bfloat16精度
        if hasattr(self.model, 'dtype') and self.model.dtype == torch.bfloat16:
            logger.info("将图模型转换为bfloat16精度以匹配主模型")
            # 直接转换整个模型
            self.graph_model = self.graph_model.to(torch.bfloat16)
            # 手动设置 dtype 属性以供检查
            # self.graph_model.dtype = torch.bfloat16
        
        # 如果有多个GPU，确保图模型也放在主GPU上
        if torch.cuda.device_count() > 1:
            logger.info(f"检测到 {torch.cuda.device_count()} 个GPU，将图模型放在 cuda:0 上")
            self.graph_model = self.graph_model.to("cuda:0")

        # TensorBoard日志设置
        self.tensorboard_dir = os.path.join(self.output_dir, "tensorboard_logs")
        self.writer = None  # 在train方法中初始化，只有使用时才创建

        # 初始化优化器
        self._init_optimizer()

    def _init_optimizer(self):
        """初始化优化器（学习率调度器将在train方法中初始化）"""
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
        
        # 学习率调度器将在train方法中初始化，因为需要知道总训练步数

    def _init_lr_scheduler(self, num_training_steps):
        """初始化学习率调度器，实现Warm-up + cosine decay策略"""
        warmup_steps = int(num_training_steps * self.warmup_ratio)  # 使用5-10%的步数作为warm-up
        logger.info(f"初始化学习率调度器: Warm-up + cosine decay, 总步数={num_training_steps}, warmup步数={warmup_steps}")
        
        # 使用cosine学习率调度器，它会在warmup后按余弦函数衰减
        self.lr_scheduler = get_scheduler(
            name="cosine",  # 改为cosine实现warm-up + cosine decay
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        logger.info(f"已创建 Warm-up + cosine decay 学习率调度器，warmup步数: {warmup_steps}")

    # --- 半监督辅助函数 ---
    def _update_adaptive_thresholds(self, Q_u):
        """
        更新自适应阈值 (内部方法)

        参数:
            Q_u: 无标签数据的预测概率 [batch_size, num_classes]
        """
        # 确保在同一设备上操作
        device = self.tau.device
        Q_u = Q_u.to(device)
        
        # 1. 计算每个样本的最大预测概率
        max_probs, _ = torch.max(Q_u, dim=1)  

        # 2. 更新全局置信度阈值（EMA）
        # tau: 通过指数移动平均(EMA)方式更新的全局置信度阈值
        # 当前批次中所有样本最大预测概率的平均值用于调整全局阈值
        batch_avg_confidence = torch.mean(max_probs)
        self.tau = self.lambda_ema * self.tau + (1 - self.lambda_ema) * batch_avg_confidence

        # 3. 更新类别平均预测概率（EMA）
        # p_tilde: 跟踪每个类别的平均预测概率，用于后续计算类别特定阈值
        # 体现了当前数据集中不同类别的分布情况
        batch_class_avg = torch.mean(Q_u, dim=0)
        self.p_tilde = self.lambda_ema * self.p_tilde + (1 - self.lambda_ema) * batch_class_avg

        # 4. 计算每个类别的自适应阈值
        # tau_c: 类别特定的置信度阈值，对稀有类别的阈值降低，常见类别的阈值提高
        # 通过将全局阈值tau与归一化的类别分布相乘来实现
        max_class_prob = torch.max(self.p_tilde)
        # 防止除以零
        self.tau_c = (self.p_tilde / (max_class_prob + 1e-8)) * self.tau

        # 限制阈值在(0, 1)之间，确保数值稳定性
        self.tau_c = torch.clamp(self.tau_c, min=1e-8, max=1.0 - 1e-8)
        self.tau = torch.clamp(self.tau, min=1e-8, max=1.0 - 1e-8)

        return self.tau, self.p_tilde, self.tau_c

    def _generate_pseudo_labels(self, Q_u):
        """
        生成伪标签 (内部方法)，使用S1优势差值(Δ-Margin)策略
        
        当多个类别都超过各自阈值时，计算各自的超阈差值(概率-阈值)，
        只保留差值最大的类别，且该最大差值必须大于容忍边界ε
        
        参数:
            Q_u: 无标签数据的预测概率 [batch_size, num_classes]
            
        返回:
            P_u: 生成的伪标签 [batch_size, num_classes] (0.0或1.0)
            mask: 有效伪标签的掩码 [batch_size] (0.0或1.0)
        """
        # 确保tau_c在与Q_u相同的设备上
        device = Q_u.device
        tau_c = self.tau_c.to(device)
        margin_epsilon = torch.tensor(self.margin_epsilon, device=device)
        
        # 使用tau_c (类别特定阈值) 检查哪些样本超过阈值
        tau_c_expanded = tau_c.unsqueeze(0)  # [1, num_classes]
        above_threshold = (Q_u > tau_c_expanded)  # [batch_size, num_classes] 布尔类型
        
        # 计算每个类别的超阈差值 (预测概率 - 阈值)
        margin_values = Q_u - tau_c_expanded  # [batch_size, num_classes]
        
        # 将未超过阈值的类别差值设为负无穷，确保不会被选择
        margin_values = torch.where(above_threshold, margin_values, 
                                  torch.tensor(-float('inf'), device=device))
        
        # 创建全零的伪标签矩阵
        P_u = torch.zeros_like(Q_u)
        
        # 对于每个样本，找出最大超阈差值的类别
        for i in range(Q_u.shape[0]):
            # 获取当前样本所有超过阈值的类别
            valid_classes = torch.where(above_threshold[i])[0]
            
            if len(valid_classes) > 0:
                # 找出差值最大的类别
                max_margin_idx = torch.argmax(margin_values[i]).item()
                max_margin_value = margin_values[i, max_margin_idx].item()
                
                # 只有当最大差值超过容忍边界时，才生成伪标签
                if max_margin_value > margin_epsilon.item():
                    P_u[i, max_margin_idx] = 1.0
        
        # 创建掩码：有任何一个类别被标记为1的样本
        mask = (torch.sum(P_u, dim=1) > 0).float()

        return P_u, mask

    def _compute_losses(self, Y_l, L_l):
        """
        计算监督损失，将损失计算移到CPU上
        
        参数:
            Y_l: 有标签数据的预测 logits [batch_size_l, num_classes]
            L_l: 有标签数据的真实标签 [batch_size_l] (索引)
            
        返回:
            total_loss: 总损失 (即监督损失)
            sup_loss: 监督损失 (与total_loss相同)
            pseudo_loss: 为兼容性保留，值为0
        """
        # 判断输入的有效性
        if Y_l is None or L_l is None:
            # 如果任一输入为None，返回零损失
            zero_loss = torch.tensor(0.0, device="cpu")
            return zero_loss, zero_loss, zero_loss
            
        # 确保预测概率和标签张量都有数据
        if Y_l.nelement() == 0 or L_l.nelement() == 0:
            zero_loss = torch.tensor(0.0, device="cpu")
            return zero_loss, zero_loss, zero_loss
        
        # 检查并过滤无效标签 (-1或不在有效范围内的标签)
        valid_labeled_indices = (L_l != -1) & (L_l >= 0) & (L_l < self.num_classes)
        
        # 初始化损失为0
        sup_loss = torch.tensor(0.0, device="cpu")
        
        # 仅当有有效标签时才计算损失
        if torch.any(valid_labeled_indices):
            # 将数据移动到CPU计算损失
            Y_l_cpu = Y_l[valid_labeled_indices].float().cpu()
            L_l_cpu = L_l[valid_labeled_indices].long().cpu()
            
            # 确保输入有效 - 避免空张量
            if Y_l_cpu.size(0) > 0:
                criterion = nn.CrossEntropyLoss()
                sup_loss = criterion(Y_l_cpu, L_l_cpu)
        
        # 总损失就是监督损失
        total_loss = sup_loss

        # 将损失移回原设备以便反向传播
        if Y_l.device.type == "cuda":
            total_loss = total_loss.to(Y_l.device)
            sup_loss = sup_loss.to(Y_l.device)

        return total_loss, sup_loss, torch.tensor(0.0, device=sup_loss.device)  # 返回零伪标签损失

    def _update_labeled_dataset(
        self,
        train_dataset,
        labeled_indices,
        unlabeled_indices,
        accelerator
    ):
        """使用高置信度伪标签更新训练数据集，按类别选择前K%的样本 (优化内存占用)"""

        logger.info("开始更新有标签数据集 (内存优化版)...")

        # 切换到评估模式
        self.model.eval()
        self.graph_model.eval()

        # 创建临时数据加载器处理所有无标签数据
        temp_unlabeled_dataset = torch.utils.data.Subset(train_dataset, unlabeled_indices)


        temp_unlabeled_loader = DataLoader(
            temp_unlabeled_dataset,
            batch_size=1, # 使用获取的或默认的 batch_size
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            num_workers=4,
            shuffle=False # 更新时不应打乱顺序，以便索引对应
        )

        # 准备临时加载器
        temp_unlabeled_loader = accelerator.prepare(temp_unlabeled_loader)

        # 收集所有有效的伪标签候选信息 (轻量级元组)
        all_pseudo_label_candidates = [] # 存储 (confidence, original_index, pseudo_label_class)

        logger.info(f"处理 {len(unlabeled_indices)} 个无标签样本生成伪标签...")

        with torch.no_grad():
            update_progress_bar = tqdm(temp_unlabeled_loader, desc="生成伪标签用于更新数据集")

            for batch in update_progress_bar:
                inputs_dict = self._prepare_batch_inputs(batch)
                # 修改：从phone_ids获取原始索引 - 这一步逻辑保持不变，但需要确保正确性
                phone_ids_batch = batch['phone_ids'] # 获取这批数据的phone_ids

                # 获取原始索引 (需要优化查找效率, 但暂时保持原逻辑)
                original_batch_indices = []
                # 尝试从 batch 获取在 subset 中的索引，如果 collate_fn 没有提供，则生成 range
                current_batch_indices_in_subset = batch.get('indices_in_subset', range(1 * accelerator.num_processes))
                actual_batch_size = len(phone_ids_batch) # 获取当前批次的实际大小

                # 确保 current_batch_indices_in_subset 长度与实际批次大小匹配
                if len(current_batch_indices_in_subset) < actual_batch_size:
                     logger.warning(f"Indices in subset length ({len(current_batch_indices_in_subset)}) mismatch with actual batch size ({actual_batch_size}). Using range.")
                     current_batch_indices_in_subset = range(actual_batch_size)
                elif len(current_batch_indices_in_subset) > actual_batch_size:
                     current_batch_indices_in_subset = current_batch_indices_in_subset[:actual_batch_size]

                for sub_idx in current_batch_indices_in_subset:
                    # 确保 sub_idx 在 unlabeled_indices 的有效范围内
                    if sub_idx < len(unlabeled_indices):
                        original_idx = unlabeled_indices[sub_idx] # 从全局unlabeled_indices映射回原始索引
                        original_batch_indices.append(original_idx)
                    else:
                        # 如果索引超出范围（可能发生在最后一个批次且大小不一致时），记录警告并跳过
                        logger.warning(f"Subset index {sub_idx} out of bounds for unlabeled_indices (len: {len(unlabeled_indices)}). Skipping.")
                        # 可以选择添加一个 placeholder 或直接跳过，这里选择跳过对应的原始索引查找
                        # 这意味着后续如果这个位置有伪标签，也无法关联到原始索引
                        pass # 或者 original_batch_indices.append(None) 并处理

                # 获取预测概率
                outputs = self.model.thinker(**inputs_dict)
                probs = self.get_class_probabilities(outputs)

                # 生成伪标签
                P_u_batch, mask_batch = self._generate_pseudo_labels(probs)

                # 计算置信度 (最大预测概率)
                max_probs, _ = torch.max(probs, dim=1)

                # 收集有效的伪标签候选者
                valid_mask = mask_batch.bool()
                valid_indices_in_batch = torch.where(valid_mask)[0]

                # 确保索引在正确的设备上
                valid_indices_cpu = valid_indices_in_batch.cpu().numpy()
                
                for j in valid_indices_cpu: # 迭代有效索引
                    if j < len(original_batch_indices): # 确保索引有效且原始索引已找到
                        pseudo_label_class = torch.argmax(P_u_batch[j]).item()
                        confidence = max_probs[j].item()
                        original_idx = original_batch_indices[j]
                        if original_idx is not None: # 再次检查 original_idx 是否有效
                           # 收集轻量级信息
                           all_pseudo_label_candidates.append((confidence, original_idx, pseudo_label_class))
                    elif j < actual_batch_size: # 如果索引有效但原始索引查找失败或跳过
                         logger.warning(f"Valid pseudo-label found at batch index {j}, but could not map to original dataset index. Skipping candidate.")

                # 清理批次内的内存
                del inputs_dict, outputs, probs, P_u_batch, mask_batch, max_probs, valid_mask, valid_indices_in_batch
                torch.cuda.empty_cache()

        # --- 后处理：对收集到的所有候选者进行分组、排序和选择 ---
        logger.info(f"收集到 {len(all_pseudo_label_candidates)} 个有效的伪标签候选者，开始按类别选择 Top K%...")

        # 按类别分组
        class_grouped_samples = {c: [] for c in range(self.num_classes)}
        for confidence, original_idx, pseudo_label_class in all_pseudo_label_candidates:
            class_grouped_samples[pseudo_label_class].append({
                'confidence': confidence,
                'index': original_idx
            })

        # 清理候选列表，释放内存
        del all_pseudo_label_candidates
        import gc
        gc.collect()

        # 新的选择逻辑：从每个类别中选择前K%的样本
        newly_labeled_indices = []

        for class_id, samples in class_grouped_samples.items():
            if samples: # 如果这个类别有样本
                # 按置信度排序
                sorted_samples = sorted(samples, key=lambda x: x['confidence'], reverse=True)

                # 计算要选择的样本数量（每个类别的前K%）
                # 确保至少选择一个，如果 K% 结果小于 1
                num_to_select = max(1, int(len(sorted_samples) * self.top_k_percent))
                # 同时确保不超过该类别的样本总数
                num_to_select = min(num_to_select, len(sorted_samples))

                selected_samples = sorted_samples[:num_to_select]

                # 记录统计信息
                logger.info(f"类别 {class_id}: 从 {len(sorted_samples)} 个伪标签中选择 {num_to_select} 个")

                # 更新数据集
                for sample in selected_samples:
                    orig_idx = sample['index']
                    try:
                        # 直接修改 train_dataset.data 中的标签
                        # 注意：这会直接修改原始数据集对象
                        train_dataset.data[orig_idx]['label'] = class_id
                        newly_labeled_indices.append(orig_idx)
                    except (TypeError, IndexError, KeyError) as e:
                        logger.error(f"更新索引 {orig_idx} 的标签失败: {e}，跳过此样本")
                        continue

        # 更新索引列表
        if newly_labeled_indices:
            # 从 unlabeled_indices 中移除新标记的索引
            # 使用集合操作提高效率
            newly_labeled_set = set(newly_labeled_indices)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in newly_labeled_set]
            # 将新标记的索引添加到 labeled_indices
            labeled_indices.extend(newly_labeled_indices)
            logger.info(f"数据集已更新。新增标签: {len(newly_labeled_indices)} 个。总标签: {len(labeled_indices)} 个，剩余无标签: {len(unlabeled_indices)} 个")
        else:
            logger.warning("没有样本满足添加到有标签集的条件")

        # 清理临时加载器和恢复训练模式
        # 使用locals()查询变量是否存在，防止删除未定义变量引发的错误
        delete_vars = ['temp_unlabeled_dataset', 'temp_unlabeled_loader', 'class_grouped_samples']
        to_delete = [var for var in delete_vars if var in locals()]
        
        # 添加可能不存在的变量（取决于代码执行路径）
        for var in ['samples', 'sorted_samples', 'selected_samples']:
            if var in locals():
                to_delete.append(var)
                
        # 安全删除变量
        for var in to_delete:
            del locals()[var]
            
        torch.cuda.empty_cache()
        self.model.train()
        self.graph_model.train()

        return labeled_indices, unlabeled_indices, newly_labeled_indices

    def get_class_probabilities(self, outputs):
        """获取最后一个位置4个标签各自的概率，减少中间变量"""
        # 直接从输出获取最后一个位置的 logits
        last_token_logits = outputs.logits[:, -1, :]
        
        # 获取选项对应的 token_ids
        option_tokens = [
            self.processor.tokenizer.convert_tokens_to_ids("A"),
            self.processor.tokenizer.convert_tokens_to_ids("B"),
            self.processor.tokenizer.convert_tokens_to_ids("C"),
            self.processor.tokenizer.convert_tokens_to_ids("D")
        ]
        
        # 只提取需要的 token 的 logits
        class_logits = torch.stack([last_token_logits[:, token_id] for token_id in option_tokens], dim=1)
        
        # 应用 softmax 获取概率
        probability_scores = F.softmax(class_logits, dim=1)
        
        # 显式释放中间变量
        del last_token_logits, class_logits
        
        return probability_scores
    
    def train(
        self,
        train_data_path,
        num_epochs=5,
        gradient_accumulation_steps=1,
        log_interval=5,
        use_tensorboard=False,
        batch_size=1,
        use_balanced_sampling=True,
        save_callback=None,  # 添加保存回调函数参数
        # --- 添加断点重连参数 ---
        checkpoint_freq=2,  # 每隔几个epoch保存一次检查点
        resume_from_checkpoint=None,  # 从指定检查点恢复训练
        save_checkpoints=True  # 是否保存检查点
    ):
        """
        训练模型，支持半监督学习和断点重连

        参数:
            train_data_path: 有标签训练数据CSV文件所在的目录
            num_epochs: 训练轮数
            gradient_accumulation_steps: 梯度累积步数
            log_interval: 日志记录间隔
            use_tensorboard: 是否使用tensorboard记录实验
            batch_size: 批次大小
            use_balanced_sampling: 是否使用平衡采样
            save_callback: 模型保存后的回调函数，接收保存路径作为参数
            checkpoint_freq: 每隔几个epoch保存一次检查点
            resume_from_checkpoint: 从指定检查点恢复训练，可以是路径或者None
            save_checkpoints: 是否保存检查点
        """
        if self.use_semi_supervised:
             logger.info("启用半监督学习模式。")

        # 初始化TensorBoard
        if use_tensorboard:  # 参数名保持不变，实际是使用tensorboard
            # 创建tensorboard日志目录
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
            
            # 记录配置信息到tensorboard
            self.writer.add_text("训练配置/学习率", str(self.learning_rate))
            self.writer.add_text("训练配置/权重衰减", str(self.weight_decay))
            self.writer.add_text("训练配置/批次大小", str(batch_size))
            self.writer.add_text("训练配置/梯度累积步数", str(gradient_accumulation_steps))
            self.writer.add_text("训练配置/训练轮数", str(num_epochs))
            self.writer.add_text("训练配置/使用LoRA", str(self.use_lora))
            
            if self.use_lora:
                self.writer.add_text("LoRA配置/rank", str(self.lora_r))
                self.writer.add_text("LoRA配置/alpha", str(self.lora_alpha))
                self.writer.add_text("LoRA配置/dropout", str(self.lora_dropout))
            
            if self.use_semi_supervised:
                self.writer.add_text("半监督配置/EMA系数", str(self.lambda_ema))
                self.writer.add_text("半监督配置/初始阈值", str(self.initial_tau))
                self.writer.add_text("半监督配置/top_k百分比", str(self.top_k_percent))
                self.writer.add_text("半监督配置/更新频率", str(self.update_dataset_freq))
                
            # 添加学习率调度器配置信息
            self.writer.add_text("学习率调度器/类型", str(self.lr_scheduler_type))
            self.writer.add_text("学习率调度器/Warmup比例", str(self.warmup_ratio))
                
            logger.info(f"TensorBoard日志将保存在: {self.tensorboard_dir}")
            logger.info("可以使用 'tensorboard --logdir=你的日志目录' 命令查看训练过程")

        # 创建训练数据集和数据加载器
        train_labels_file = 'train_filtered.csv'
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
        
        # 创建一个统一的数据加载器，使用LabelBalancedSampler确保有标签数据分布合理
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_config.batch_size,
            collate_fn=train_dataset.collate_fn,
            sampler=LabelBalancedSampler(train_dataset, batch_size=batch_size) if use_balanced_sampling else None,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True
        )

        accelerator = Accelerator()
        # 准备模型、优化器和训练加载器
        self.model, self.optimizer, train_loader = accelerator.prepare(
            self.model, self.optimizer, train_loader
        )

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 训练循环
        global_step = 0
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        
        # 初始化学习率调度器
        if self.num_training_steps is None:
            self.num_training_steps = total_steps
            
        self._init_lr_scheduler(self.num_training_steps)
        
        # 如果使用学习率调度器，将其传递给accelerator
        if self.lr_scheduler is not None:
            self.lr_scheduler = accelerator.prepare(self.lr_scheduler)
            logger.info(f"学习率调度器已初始化，总训练步数: {self.num_training_steps}")
            
            # 记录初始学习率到TensorBoard
            if use_tensorboard and self.writer:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    current_lr = param_group['lr']
                    self.writer.add_scalar(f"train/initial_lr_group_{i}", current_lr, 0)
                    logger.info(f"初始学习率(组 {i}): {current_lr}")
        
        # 确定起始epoch和状态
        start_epoch = 0
        
        # 如果指定了恢复检查点，尝试加载
        if resume_from_checkpoint:
            logger.info(f"尝试从检查点 {resume_from_checkpoint} 恢复训练...")
            try:
                # 加载检查点
                self.load_model(resume_from_checkpoint)
                
                # 尝试确定起始epoch
                checkpoint_name = os.path.basename(resume_from_checkpoint)
                if checkpoint_name.startswith("checkpoint-"):
                    try:
                        start_epoch = int(checkpoint_name.split("-")[1])
                        logger.info(f"从epoch {start_epoch} 恢复训练")
                    except (ValueError, IndexError):
                        logger.warning(f"无法从检查点名称确定起始epoch，将从0开始")
                
                # 如果已经达到总轮数，则不再训练
                if start_epoch >= num_epochs:
                    logger.info(f"检查点epoch {start_epoch} 已达到或超过总训练轮数 {num_epochs}，跳过训练")
                    return
                
                # 加载成功
                logger.info(f"成功从检查点恢复模型和状态，将从epoch {start_epoch} 继续训练")
            except Exception as e:
                logger.error(f"从检查点恢复失败: {e}，将从头开始训练")
                import traceback
                traceback.print_exc()
                start_epoch = 0

        # --- 在训练开始前重置/初始化阈值 ---
        if self.use_semi_supervised:
            # 确定目标设备 (与 __init__ 逻辑一致)
            target_device = self.device
            if hasattr(self.model, 'hf_device_map'):
                try:
                    target_device = next(self.model.parameters()).device
                except StopIteration:
                    target_device = self.device
            elif hasattr(self.model, 'device'):
                target_device = self.model.device
            
            logger.info(f"Resetting semi-supervised thresholds on device: {target_device}")
            self.tau = torch.tensor(self.initial_tau, device=target_device)
            self.p_tilde = torch.ones(self.num_classes, device=target_device) / self.num_classes
            self.tau_c = self.tau * (self.p_tilde / torch.max(self.p_tilde))

        for epoch in range(start_epoch, num_epochs):
            # === 训练模型 ===
            self.model.train()
            self.graph_model.train()

            epoch_loss = 0
            epoch_sup_loss = 0
            num_batches = 0
            last_batch_loss = None  # Variable to store the last computed batch loss
            
            # 处理所有数据
            train_progress_bar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{num_epochs} (训练中)"
            )
            
            for batch_idx, batch in enumerate(train_progress_bar):
                num_batches += 1
                
                # 准备输入
                inputs_dict = self._prepare_batch_inputs(batch)
                
                # 获取模型的设备
                model_device = next(self.model.parameters()).device
                
                # --- 处理批次标签 ---
                raw_labels = batch['label']
                valid_labels_list = []
                
                # 遍历批次中的标签，确定哪些是有效的（不等于-1）
                for lbl in raw_labels:
                    if lbl is not None:
                        try:
                            valid_labels_list.append(int(lbl))
                        except (ValueError, TypeError):
                            valid_labels_list.append(-1)
                    else:
                        valid_labels_list.append(-1)
                
                # 将处理后的标签转换为张量
                labels = torch.tensor(valid_labels_list, dtype=torch.long, device=model_device)
                
                # 区分有标签和无标签样本
                labeled_mask = labels != -1
                unlabeled_mask = ~labeled_mask
                
                # 前向传播（所有样本）
                # print("phone_ids: ", batch['phone_ids'])
                outputs = self.model.thinker(**inputs_dict)
                
                
                
                probabilities = self.get_class_probabilities(outputs)
                
                # 计算损失（仅有标签样本）
                if torch.any(labeled_mask):
                    # 确保掩码在与概率相同的设备上
                    labeled_mask_device = labeled_mask.to(probabilities.device)

                    # 使用正确设备上的掩码
                    Y_l = probabilities[labeled_mask_device]
                    L_l = labels[labeled_mask].to(probabilities.device)
                    
                    # 计算监督损失 - 使用_compute_losses方法替代直接使用CrossEntropyLoss
                    current_batch_loss, sup_loss, _ = self._compute_losses(Y_l, L_l)
                    last_batch_loss = current_batch_loss.item()
                    
                    # Scale loss for gradient accumulation
                    loss_scaled = current_batch_loss / gradient_accumulation_steps
                    accelerator.backward(loss_scaled)
                    
                    # Accumulate unscaled loss for epoch average
                    epoch_loss += last_batch_loss
                    epoch_sup_loss += sup_loss.item() if sup_loss is not None else 0.0
                    
                    del Y_l, L_l, current_batch_loss, sup_loss, loss_scaled
                    torch.cuda.empty_cache()
                
                # 处理无标签样本（仅在启用半监督学习时）
                if self.use_semi_supervised and torch.any(unlabeled_mask):
                    # 不计算梯度，仅更新阈值
                    with torch.no_grad():
                        # 重新确保掩码在正确的设备上（防止之前的修改被覆盖）
                        unlabeled_mask_device = unlabeled_mask.to(probabilities.device)
                        Q_u = probabilities[unlabeled_mask_device]

                        # 更新自适应阈值
                        self._update_adaptive_thresholds(Q_u.to(self.tau.device))
                
                        del Q_u, unlabeled_mask_device
                        torch.cuda.empty_cache()

                # 在梯度累积和优化器步骤后清理内存
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx + 1 == len(train_loader):
                    self.optimizer.step()
                    
                    # 更新学习率调度器
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                        
                        # 记录当前学习率
                        if use_tensorboard and self.writer and global_step % log_interval == 0:
                            for i, param_group in enumerate(self.optimizer.param_groups):
                                current_lr = param_group['lr']
                                self.writer.add_scalar(f"train/lr_group_{i}", current_lr, global_step)
                    
                    self.optimizer.zero_grad()
                    
                    # --- Log step-level metrics HERE (after optimizer step) ---
                    if use_tensorboard and self.writer and global_step % log_interval == 0:
                        if last_batch_loss is not None:
                            # Log the loss of the *last batch* that contributed to this optimizer step
                            self.writer.add_scalar("train/batch_supervised_loss", last_batch_loss, global_step)
                    
                        # Reset last_batch_loss to avoid logging stale values if subsequent steps have no labeled data
                        last_batch_loss = None
                            
                    
                    torch.cuda.empty_cache()

                # 更新全局步数
                global_step += 1
                
                # 清理内存
                self._clean_memory(
                    inputs_dict, outputs, probabilities,
                    locals().get('current_batch_loss'),
                    locals().get('loss_scaled'),
                    locals().get('Y_l'),
                    locals().get('L_l'),
                    locals().get('labeled_mask_device'),
                    locals().get('unlabeled_mask')
                )
        
            # === Epoch结束，打印统计信息 ===
            avg_epoch_loss = epoch_loss / max(num_batches, 1) # Use max to avoid division by zero
            avg_epoch_sup_loss = epoch_sup_loss / max(num_batches, 1)

            logger.info(f"Epoch {epoch+1}/{num_epochs} completed:")
            logger.info(f"  Average Epoch Loss: {avg_epoch_loss:.4f}") # Renamed for clarity
            logger.info(f"  Average Epoch Supervised Loss: {avg_epoch_sup_loss:.4f}") # Renamed for clarity
            if self.use_semi_supervised:
                logger.info(f"  全局置信度阈值 (tau): {self.tau.item():.4f}")
                # Ensure tau_c and p_tilde are on CPU for printing if they aren't already
                tau_c_cpu = self.tau_c.cpu() if self.tau_c.is_cuda else self.tau_c
                p_tilde_cpu = self.p_tilde.cpu() if self.p_tilde.is_cuda else self.p_tilde
                logger.info(f"  类别特定阈值 (tau_c): {tau_c_cpu}")
                logger.info(f"  类别分布估计 (p_tilde): {p_tilde_cpu}")

            if use_tensorboard and self.writer:
                # 记录epoch级别的指标 (Keep epoch/supervised_loss)
                self.writer.add_scalar("epoch/supervised_loss", avg_epoch_sup_loss, epoch + 1)
                # self.writer.add_scalar("epoch/total_loss", avg_epoch_loss, epoch + 1) # Optional: log average epoch total loss if needed

                if self.use_semi_supervised:
                    self.writer.add_scalar("epoch/global_threshold", self.tau.item(), epoch + 1)
                    # 不再记录平均值，而是为每个类别单独记录阈值
                    # self.writer.add_scalar("epoch/class_threshold_mean", self.tau_c.mean().item(), epoch + 1)
                    # self.writer.add_scalar("epoch/class_distribution_mean", self.p_tilde.mean().item(), epoch + 1)
                    
                    # 为每个类别单独记录阈值和分布
                    for class_idx in range(self.num_classes):
                        self.writer.add_scalar(f"epoch/class_{class_idx}_threshold", self.tau_c[class_idx].item(), epoch + 1)
                        self.writer.add_scalar(f"epoch/class_{class_idx}_distribution", self.p_tilde[class_idx].item(), epoch + 1)
            
            # === 保存检查点 ===
            if save_checkpoints and (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{epoch+1}")
                logger.info(f"保存检查点到: {checkpoint_path}")
                
                try:
                    self.save_model(checkpoint_path)
                    # 如果提供了回调函数，则调用它
                    if save_callback is not None:
                        save_callback(checkpoint_path)
                    logger.info(f"检查点保存成功")
                except Exception as e:
                    logger.error(f"保存检查点时出错: {e}")
                    import traceback
                    traceback.print_exc()

            # === 更新数据集（使用高置信度伪标签） ===
            if self.use_semi_supervised and unlabeled_indices and \
               (epoch + 1) % self.update_dataset_freq == 0 and epoch < num_epochs - 1:
                
                logger.info(f"Epoch {epoch+1}: 使用伪标签更新数据集...")
                
                # 更新有标签和无标签索引
                labeled_indices = []
                unlabeled_indices = []
                for i, item in enumerate(train_dataset.data):
                    if item.get('label') is not None and str(item.get('label')).strip():
                        labeled_indices.append(i)
                    else:
                        unlabeled_indices.append(i)
                
                # 使用_update_labeled_dataset函数更新数据集
                labeled_indices, unlabeled_indices, newly_labeled_indices = self._update_labeled_dataset(
                    train_dataset,
                    labeled_indices,
                    unlabeled_indices,
                    accelerator
                )
                
                # 如果有新增的有标签样本，将其写入标签文件
                if newly_labeled_indices:
                    logger.info(f"将{len(newly_labeled_indices)}个高置信度伪标签写入标签文件...")
                    self._update_labels_file(train_dataset, newly_labeled_indices, train_data_path, train_labels_file)
                    
                    # 重新加载数据集和数据加载器
                    logger.info("重新加载数据集和数据加载器...")
                    train_dataset = AudioSegmentDataset(
                        data_path=train_config.data_path,
                        model_path=train_config.model_path,
                        data_processor=self.processor,
                        labels_file=train_config.labels_file,
                        cache_dir=train_config.cache_dir,
                    )
                    
                    # 重新创建数据加载器
                    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=train_config.batch_size,
                        collate_fn=train_dataset.collate_fn,
                        sampler=LabelBalancedSampler(train_dataset, batch_size=batch_size) if use_balanced_sampling else None,
                        pin_memory=True,
                        num_workers=4,
                        persistent_workers=True
                    )
                    
                    # 准备数据加载器
                    train_loader = accelerator.prepare(train_loader)
                    
                    logger.info("数据加载器已重新创建")
            
        # 保存最终模型 (保留这部分)
        final_model_path = os.path.join(self.output_dir, "final-model")
        self.save_model(final_model_path)
        # 如果提供了回调函数，则调用它
        if save_callback is not None:
            save_callback(final_model_path) # 对于final model，回调仍然可能有用（例如日志记录）

        if use_tensorboard and self.writer:
            self.writer.close()
            logger.info(f"TensorBoard日志已保存在 {self.tensorboard_dir}")
            logger.info("可以使用 'tensorboard --logdir=你的日志目录' 命令查看训练结果")

    def _prepare_batch_inputs(self, batch):
        """将批次数据移动到设备并准备模型输入字典 (内部方法)"""
        speakers = batch['speakers']
        # 确定tensors应该放在哪个设备上(针对DataParallel)
        # 使用模型的第一个模块所在的设备
        model_device = next(self.model.parameters()).device
        
        # Move to device
        segment_features = batch['segment_features'].to(model_device) if batch['segment_features'] is not None else None
        segment_attention_mask = batch['segment_attention_mask'].to(model_device) if batch['segment_attention_mask'] is not None else None
        input_ids = batch['input_ids'].to(model_device) if batch['input_ids'] is not None else None
        input_features = batch['input_features'].to(model_device) if batch['input_features'] is not None else None
        # --- 添加类型转换 ---
        if input_features is not None and hasattr(self.model.thinker, 'dtype') and self.model.thinker.dtype == torch.bfloat16:
            input_features = input_features.to(torch.bfloat16)
            
        # --- 转换 segment_features ---
        if segment_features is not None and hasattr(self.graph_model, 'dtype') and self.graph_model.dtype == torch.bfloat16:
            segment_features = segment_features.to(torch.bfloat16)

        # print(f"input_features: {input_features.shape}") # 保持你添加的打印语句（如果需要）
        feature_attention_mask = batch['feature_attention_mask'].to(model_device) if batch['feature_attention_mask'] is not None else None
        attention_mask = batch['attention_mask'].to(model_device) if batch['attention_mask'] is not None else None


        # 使用图网络处理segment_features
        graph_feature = None
        if segment_features is not None and segment_features.nelement() > 0: # 检查非空
             try:
                 graph_feature = self.graph_model(
                     segment_features,
                     speakers,
                     attention_masks=segment_attention_mask
                 )
             except Exception as e:
                 logger.warning(f"图网络处理失败: {e}")
                 # 打印出错误的电话号码和相关张量形状信息
                 batch_indices = batch.get('index', list(range(len(batch['speakers']))))
                 phone_numbers = batch.get('phone_number', ['未知'] * len(batch['speakers']))
                 
                 logger.error(f"图网络处理失败详情:")
                 for i, (idx, phone) in enumerate(zip(batch_indices, phone_numbers)):
                     logger.error(f"样本索引: {idx}, 电话号码: {phone}")
                     
                 logger.error(f"张量形状信息:")
                 logger.error(f"segment_features形状: {segment_features.shape}")
                 logger.error(f"speakers长度: {len(speakers)}")
                 if segment_attention_mask is not None:
                     logger.error(f"segment_attention_mask形状: {segment_attention_mask.shape}")
                 
                 # 检查是否有NaN或无穷大值
                 if torch.isnan(segment_features).any() or torch.isinf(segment_features).any():
                    logger.error(f"segment_features中包含NaN或无穷大值")
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

    def test(self, test_loader, use_tensorboard=False):
        """
        在测试集上评估最终模型 (适配多GPU)

        参数:
            test_loader: 测试数据加载器
            use_tensorboard: 是否使用tensorboard记录测试结果

        返回:
            测试结果字典
        """
        logger.info("Starting final evaluation on the test set...")
        self.model.eval()
        self.graph_model.eval()

        # 将test_loader也通过accelerator准备，确保其适用于分布式环境
        accelerator = Accelerator()
        test_loader = accelerator.prepare(test_loader)

        test_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []

        num_loss_batches = 0 # Track batches where loss was computed

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                num_batches += 1

                try:
                    # 使用相同的批次处理逻辑
                    inputs_dict = self._prepare_batch_inputs(batch)
                    
                    # 获取模型设备
                    model_device = next(self.model.parameters()).device
                    
                    # --- 统一标签处理逻辑 ---
                    raw_labels = batch['label']
                    valid_labels_list = []
                    
                    # 遍历批次中的标签，确定哪些是有效的（不等于-1）
                    for lbl in raw_labels:
                        if lbl is not None:
                            try:
                                valid_labels_list.append(int(lbl))
                            except (ValueError, TypeError):
                                valid_labels_list.append(-1)
                        else:
                            valid_labels_list.append(-1)
                    
                    # 将处理后的标签转换为张量
                    labels = torch.tensor(valid_labels_list, dtype=torch.long, device=model_device)
                    
                    # 区分有标签和无标签样本
                    labeled_mask = labels != -1
                    
                    # 前向传播（所有样本）
                    outputs = self.model.thinker(**inputs_dict)
                    probabilities = self.get_class_probabilities(outputs)
                    
                    # 计算损失（仅有标签样本）
                    if torch.any(labeled_mask):
                        # 确保掩码在与概率相同的设备上
                        labeled_mask_device = labeled_mask.to(probabilities.device)
                        
                        # 使用正确设备上的掩码
                        Y_l = probabilities[labeled_mask_device]
                        L_l = labels[labeled_mask].to(probabilities.device)
                        
                        # 使用统一的损失计算方法
                        current_batch_loss, _, _ = self._compute_losses(Y_l, L_l)
                        test_loss += current_batch_loss.item()
                        num_loss_batches += 1

                    # 计算预测结果
                    preds = torch.argmax(probabilities, dim=1)
                    
                    # 使用accelerator.gather来收集分布式环境中的预测和标签
                    # 注意：accelerator.gather要求所有张量必须在同一设备上
                    # 我们将所有张量移到主设备（通常是第一个设备）
                    main_device = accelerator.device
                    
                    # 将预测和标签移到主设备
                    preds_on_main = preds.to(main_device)
                    labels_on_main = labels.to(main_device)
                    
                    # 现在在同一设备上执行gather操作
                    gathered_preds = accelerator.gather(preds_on_main)
                    gathered_labels = accelerator.gather(labels_on_main)
                    
                    # 将结果转移到CPU进行NumPy转换
                    all_preds.extend(gathered_preds.cpu().numpy())
                    all_labels.extend(gathered_labels.cpu().numpy())
                    
                    # 清理内存
                    self._clean_memory(
                        inputs_dict, outputs, probabilities, 
                        preds, labels, 
                        locals().get('current_batch_loss'),
                        locals().get('Y_l'),
                        locals().get('L_l'),
                        locals().get('labeled_mask_device')
                    )

                except Exception as e:
                    logger.error(f"测试时处理批次出错: {e}")
                    import traceback
                    traceback.print_exc()
                    # 打印更多诊断信息
                    if 'labels' in locals() and labels is not None:
                        logger.error(f"labels device: {labels.device}, shape: {labels.shape}")
                    if 'probabilities' in locals() and probabilities is not None:
                        logger.error(f"probabilities device: {probabilities.device}, shape: {probabilities.shape}")
                    if 'labeled_mask' in locals() and labeled_mask is not None:
                        logger.error(f"labeled_mask device: {labeled_mask.device}, shape: {labeled_mask.shape}")
                    
                    batch_size = labels.size(0) if 'labels' in locals() and labels is not None else 1
                    all_preds.extend([-1] * batch_size)
                    all_labels.extend([-1] * batch_size)
                    continue

        # 计算平均损失 (仅基于有标签的样本)
        avg_loss = test_loss / max(num_loss_batches, 1)

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

        # 记录测试结果 (TensorBoard)
        if use_tensorboard and results and self.writer is not None:
            for k, v in results.items():
                metric_name = k.replace('test_', '')
                self.writer.add_scalar(f"test/{metric_name}", v, 0)
            
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

        # self.processor.save_pretrained(output_path)   ### 检查
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
            "margin_epsilon": self.margin_epsilon,  # 添加伪标签生成的最小可接受超阈差值
            # 保存调度器参数
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
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
                self.margin_epsilon = loaded_config.get('margin_epsilon', 0.05)  # 加载伪标签生成的最小可接受超阈差值
                self.num_classes = loaded_config.get('num_classes', 4) # 恢复 num_classes
                
                # 加载调度器参数
                self.lr_scheduler_type = loaded_config.get('lr_scheduler_type', self.lr_scheduler_type)
                self.warmup_ratio = loaded_config.get('warmup_ratio', self.warmup_ratio)
                
                logger.info(f"半监督模式: {self.use_semi_supervised}")
                logger.info(f"伪标签超阈差值: {self.margin_epsilon}")
                logger.info(f"学习率调度器: {self.lr_scheduler_type}, Warmup比例: {self.warmup_ratio}")

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

                # 修改：不指定adapter_name，使用默认值与保存时保持一致
                self.model.load_adapter(model_path) # 移除adapter_name参数
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

        # --- 重新初始化优化器 (加载后可能需要) ---
        self._init_optimizer()
        logger.info("优化器已重新初始化。")

    def _update_labels_file(self, train_dataset, newly_labeled_indices, data_path, labels_file):
        """将高置信度伪标签写入标签文件，使用列名而非固定索引"""
        import pandas as pd
        import os
        import shutil
        
        labels_file_path = os.path.join(data_path, labels_file)
        
        try:
            # 读取原始标签文件
            try:
                df = pd.read_csv(labels_file_path)
                logger.info(f"成功读取标签文件: {labels_file_path}")
            except Exception as e:
                logger.error(f"读取标签文件失败: {e}")
                return
            
            # 创建备份
            try:
                backup_file = f"{labels_file_path}.bak"
                shutil.copy2(labels_file_path, backup_file)
                if os.path.exists(backup_file):
                    logger.info(f"已创建标签文件备份: {backup_file}")
                else:
                    logger.error(f"备份文件创建失败，文件不存在: {backup_file}")
            except Exception as e:
                logger.error(f"创建备份时出错: {str(e)}")
            
            # 确定标签列 - 查找名为'label'或包含'label'的列
            label_col_name = None
            for col in df.columns:
                if col.lower() == 'label' or 'label' in col.lower():
                    label_col_name = col
                    break
            
            # 如果没找到标签列，则默认使用最后一列
            if label_col_name is None:
                label_col_name = df.columns[-1]
                logger.warning(f"未找到明确的标签列，将使用最后一列 '{label_col_name}' 作为标签列")
            
            # 确定ID列（通常是第一列）
            id_col_name = df.columns[0]
            
            # 更新数据框中的标签
            updated_count = 0
            for idx in newly_labeled_indices:
                item = train_dataset.data[idx]
                phone_id = item['phone_id']
                pseudo_label = item['label']
                
                # 确保标签有效
                if pseudo_label is None or pseudo_label < 0 or pseudo_label >= self.num_classes:
                    logger.warning(f"样本 {phone_id} 的伪标签 {pseudo_label} 无效，跳过更新")
                    continue
                
                # 将数值标签转换为字母标签 (0->A, 1->B, 2->C, 3->D)
                label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                label_letter = label_map.get(pseudo_label)
                
                # 查找并更新对应的行
                matching_rows = df[df[id_col_name].astype(str) == str(phone_id)]
                if len(matching_rows) > 0:
                    row_idx = matching_rows.index[0]
                    df.loc[row_idx, label_col_name] = label_letter
                    updated_count += 1
                else:
                    logger.warning(f"未找到电话ID {phone_id} 对应的行，无法更新标签")
            
            # 保存更新后的文件
            if updated_count > 0:
                df.to_csv(labels_file_path, index=False)
                logger.info(f"已成功更新 {updated_count}/{len(newly_labeled_indices)} 个伪标签到标签文件")
            else:
                logger.warning("没有样本被更新，标签文件保持不变")
                
        except Exception as e:
            logger.error(f"更新标签文件时发生错误: {e}")
            import traceback
            traceback.print_exc()

    def _clean_memory(self, *tensors):
        """清理指定的张量并执行内存回收"""
        for tensor in tensors:
            if tensor is not None and hasattr(tensor, 'device') and tensor.device.type == 'cuda':
                del tensor
        
        # 清理未命名的临时变量
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main(
    batch_size=1,
    num_epochs=5,
    use_balanced_sampling=True,
    use_tensorboard=True,
    use_lora=True,
    # --- 添加半监督控制参数 ---
    use_semi_supervised=False,
    lambda_ema=0.9,
    initial_tau=0.8,
    top_k_percent=0.1,
    update_dataset_freq=1,
    margin_epsilon=0.05,    # 添加伪标签生成的最小可接受超阈差值
    # --- 添加多GPU训练参数 ---
    gradient_accumulation_steps=1,
    use_accelerator=True,
    log_interval=10,
    # --- 添加控制权重保存的参数 ---
    keep_last_n_checkpoints=2,
    save_checkpoints=True,
    # --- 添加断点重连参数 ---
    checkpoint_freq=2,
    resume_from_checkpoint=None,
    # --- 添加学习率和调度器参数 ---
    learning_rate=2e-5,              # 适当降低起始学习率
    lora_lr_factor=8,                # 对于LoRA可以稍微提高系数
    lr_scheduler_type="linear",      # 强制使用linear衰减
    warmup_ratio=0.08                # 设置为8%的warm-up
    ):
    """主函数，支持半监督学习和断点重连，并添加学习率调度策略"""
    # 配置参数
    model_path = "/data/shared/Qwen/models/Qwen2.5-Omni-7B"
    output_dir = "./outputs_semi" if use_semi_supervised else "./outputs" # 根据模式选择不同输出目录
    data_dir = "./data"  # 包含 train_filtered.csv, test.csv, audio/, segments/ 的目录
    train_data_path = data_dir # 训练数据目录
    test_data_path = data_dir  # 测试数据目录

    # 计算LoRA和全参数微调的实际学习率
    lora_learning_rate = learning_rate * lora_lr_factor if use_lora else learning_rate
    full_param_learning_rate = learning_rate
    
    actual_lr = lora_learning_rate if use_lora else full_param_learning_rate
    logger.info(f"学习率设置: 基础学习率={learning_rate}, {'LoRA学习率' if use_lora else '全参数学习率'}={actual_lr}")

    # 打印CUDA信息
    if torch.cuda.is_available():
        logger.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}, "
                       f"可用显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 尝试清理CUDA缓存
        torch.cuda.empty_cache()
        logger.info("已清空CUDA缓存")

    # 如果启用半监督，检查无标签数据路径
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
        # 修改为自动分配到多GPU
        device = "cuda"
        logger.info(f"CUDA可用，检测到 {torch.cuda.device_count()} 个GPU设备，将使用多GPU加载模型")
        try:
            model = Qwen2_5OmniTextOnlyModel.from_pretrained(
                model_path,
                config=llm_config,
                device_map="auto", # 修改为auto以使用多GPU
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
            print(f"the model dtype is {model.dtype}")
            print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else '未使用device_map'}")
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
        "speaker_embedding_dim": 16,
        "num_speakers": None, # 让模型动态处理
        "num_layers": 3,
        "dropout": 0.1,
        "similarity_threshold": 0.9,
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

    if use_lora: 
        logger.info(f"使用LoRA微调，参数: r={lora_r}, alpha={lora_alpha}, 学习率={lora_learning_rate}")
    else: 
        logger.info(f"使用全参数微调，学习率={full_param_learning_rate}")

    # 打印学习率调度器信息
    logger.info(f"学习率调度器: 类型={lr_scheduler_type}, warmup比例={warmup_ratio}")

    # 初始化训练器
    trainer = AdaptiveThresholdTrainer(
        model=model,
        processor=processor,
        graph_config=graph_config_dict,
        output_dir=output_dir,
        device=device,
        learning_rate=actual_lr,  # 使用根据LoRA计算的学习率
        weight_decay=0.01,
        num_classes=num_classes, # 传递类别数
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        # 传递半监督参数
        use_semi_supervised=use_semi_supervised,
        lambda_ema=lambda_ema,
        initial_tau=initial_tau,
        top_k_percent=top_k_percent,
        update_dataset_freq=update_dataset_freq,
        margin_epsilon=margin_epsilon,  # 传递伪标签生成的最小可接受超阈差值
        use_accelerator=use_accelerator,
        # 传递学习率调度器参数
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        num_training_steps=None  # 在train方法中动态计算
    )

    # 用于跟踪保存的checkpoint，用于清理旧的checkpoint
    saved_checkpoints = []

    # 定义保存回调函数
    def save_callback(path):
        """保存模型回调函数，用于管理checkpoint"""
        if not path.startswith(os.path.join(output_dir, "checkpoint-")):
            # 如果不是checkpoint而是epoch或final模型，则直接保存
            return
            
        # 添加checkpoint到列表中
        saved_checkpoints.append(path)
        
        # 如果列表长度超过保留数量，删除最旧的checkpoint
        if keep_last_n_checkpoints > 0 and len(saved_checkpoints) > keep_last_n_checkpoints:
            oldest_checkpoint = saved_checkpoints.pop(0)
            # 删除最旧的checkpoint目录
            try:
                logger.info(f"清理旧的checkpoint: {oldest_checkpoint}")
                import shutil
                shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            except Exception as e:
                logger.error(f"删除旧checkpoint {oldest_checkpoint} 失败: {e}")

    # 如果指定了恢复检查点，记录日志
    if resume_from_checkpoint:
        logger.info(f"将从检查点 {resume_from_checkpoint} 恢复训练...")

    # 启动训练
    trainer.train(
        train_data_path=train_data_path,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_interval=log_interval,
        use_tensorboard=use_tensorboard,
        batch_size=batch_size,
        use_balanced_sampling=use_balanced_sampling,
        save_callback=save_callback,
        checkpoint_freq=checkpoint_freq,
        resume_from_checkpoint=resume_from_checkpoint,
        save_checkpoints=save_checkpoints
    )
    
    # --- 测试阶段 ---
    logger.info("训练完成。加载最终模型进行测试...")
    final_model_path = os.path.join(output_dir, "final-model")

    try:
        trainer.load_model(final_model_path)
        logger.info(f"最终模型从 {final_model_path} 加载成功。")
    except Exception as e:
        logger.error(f"加载最终模型 {final_model_path} 失败: {e}. 无法执行测试评估。")
        if use_tensorboard and trainer.writer is not None:
            trainer.writer.close()
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
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=test_config.batch_size,
                collate_fn=test_dataset.collate_fn,
                pin_memory=True,
                num_workers=4,  # 添加多个worker以加速数据加载
                persistent_workers=True,  # 保持worker进程存活以避免频繁创建和销毁
            )
            logger.info(f"测试数据加载器创建成功，共 {len(test_dataset)} 个样本。")

            # 为测试加载器准备accelerator
            accelerator = Accelerator()
            model = trainer.model
            model, test_loader = accelerator.prepare(model, test_loader)
            
            # 临时更新模型引用然后执行测试评估
            original_model = trainer.model
            trainer.model = model
            test_results = trainer.test(test_loader, use_tensorboard=use_tensorboard)
            # 恢复原始模型引用
            trainer.model = original_model

            # 记录测试结果 (TensorBoard)
            if use_tensorboard and test_results and trainer.writer is not None:
                for k, v in test_results.items():
                    metric_name = k.replace('test_', '')
                    trainer.writer.add_scalar(f"test/{metric_name}", v, 0)  # 0作为全局步数，因为测试只进行一次

        except FileNotFoundError:
            logger.warning(f"测试标签文件 {test_labels_file} 存在但无法在 Dataset 中加载，跳过测试评估。")
        except Exception as e:
             logger.error(f"创建测试数据加载器或执行测试时出错: {e}")
             import traceback
             traceback.print_exc()
             logger.warning("跳过测试评估。")

    if use_tensorboard and trainer.writer is not None:
        trainer.writer.close()
        logger.info(f"TensorBoard日志已保存在 {trainer.tensorboard_dir}")
        logger.info("可以使用 'tensorboard --logdir=你的日志目录' 命令查看训练结果")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="音频LLM训练脚本")
    
    # 基本训练参数
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="批处理大小（默认：1）")
    parser.add_argument("--num_epochs", type=int, default=20, 
                        help="训练轮数（默认：50）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="梯度累积步数（默认：1）")
    parser.add_argument("--log_interval", type=int, default=5, 
                        help="日志记录间隔，以步数为单位（默认：5）")
    
    # 采样和优化器设置
    parser.add_argument("--use_balanced_sampling", type=lambda x: x.lower() == 'true', default=True, 
                        help="是否使用平衡采样（默认：True）")
    parser.add_argument("--use_tensorboard", type=lambda x: x.lower() == 'true', default=True, 
                        help="是否使用TensorBoard记录实验（默认：True）")
    
    # 模型和加速设置
    parser.add_argument("--use_lora", type=lambda x: x.lower() == 'true', default=True, 
                        help="是否使用LoRA微调（默认：True）")
    parser.add_argument("--use_accelerator", type=lambda x: x.lower() == 'true', default=True, 
                        help="是否使用Accelerator进行分布式训练（默认：True）")
    
    # 半监督学习参数
    parser.add_argument("--use_semi_supervised", type=lambda x: x.lower() == 'true', default=True, 
                        help="是否启用半监督学习（默认：true）")
    parser.add_argument("--lambda_ema", type=float, default=0.95, 
                        help="EMA动量参数（默认：0.95）")
    parser.add_argument("--initial_tau", type=float, default=0.9, 
                        help="初始全局阈值（默认：0.9）")
    parser.add_argument("--top_k_percent", type=float, default=0.05, 
                        help="选择伪标签样本的百分比（默认：0.05）")
    parser.add_argument("--update_dataset_freq", type=int, default=3, 
                        help="更新有标签数据集的频率，以epoch为单位（默认：3）")
    parser.add_argument("--margin_epsilon", type=float, default=0.06,
                        help="伪标签生成的最小可接受超阈差值（默认：0.06）")
    
    # 添加控制权重保存的参数
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=2,
                        help="保留最近N个检查点，其余自动删除（默认：2，设为0则保留所有）")
    parser.add_argument("--save_checkpoints", type=lambda x: x.lower() == 'true', default=True,
                        help="是否保存中间检查点（默认：True）")
    
    # 添加断点重连参数
    parser.add_argument("--checkpoint_freq", type=int, default=2,
                       help="每隔几个epoch保存一次检查点（默认：2）")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="从指定检查点恢复训练，提供检查点路径（默认：None）")
    
    # 添加学习率和调度器相关参数
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                       help="基础学习率（默认：2e-5）")
    parser.add_argument("--lora_lr_factor", type=float, default=8, 
                       help="LoRA学习率倍数，相对于基础学习率（默认：8）")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", 
                       choices=["linear", "cosine", "constant_with_warmup", "cosine_with_restarts"],
                       help="学习率调度器类型（默认：linear）")
    parser.add_argument("--warmup_ratio", type=float, default=0.12, 
                       help="warmup阶段占总步数的比例（默认：0.12）")
    
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 根据命令行参数运行主函数
    main(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        use_balanced_sampling=args.use_balanced_sampling,
        use_tensorboard=args.use_tensorboard,
        use_lora=args.use_lora,
        use_semi_supervised=args.use_semi_supervised,
        lambda_ema=args.lambda_ema,
        initial_tau=args.initial_tau,
        top_k_percent=args.top_k_percent,
        update_dataset_freq=args.update_dataset_freq,
        margin_epsilon=args.margin_epsilon,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_accelerator=args.use_accelerator,
        log_interval=args.log_interval,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        save_checkpoints=args.save_checkpoints,
        checkpoint_freq=args.checkpoint_freq,
        resume_from_checkpoint=args.resume_from_checkpoint,
        # 添加学习率和调度器参数
        learning_rate=args.learning_rate,
        lora_lr_factor=args.lora_lr_factor,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio
    )
