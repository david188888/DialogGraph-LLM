# 半监督自适应阈值微调方法

## 1. 方法概述

本文档详细介绍了一种创新的半监督自适应阈值微调方法，专为解决标注数据稀缺和类别不平衡问题而设计，尤其适用于电话营销场景中的意向分类任务。该方法通过动态调整置信度阈值，从无标签数据中挖掘高质量伪标签，同时确保各类别均能获得足够的伪标签样本。

## 2. 核心技术

### 2.1 自适应阈值机制

该系统基于三个关键参数实现自适应阈值控制：

```python
self.tau = torch.tensor(self.initial_tau, device=target_device)  # 全局置信度阈值
self.p_tilde = torch.ones(self.num_classes, device=target_device) / self.num_classes  # 类别分布估计
self.tau_c = self.tau * (self.p_tilde / torch.max(self.p_tilde))  # 类别特定阈值
```

三个参数的计算公式如下：

1. **全局置信度阈值 (τ)**：使用指数移动平均 (EMA) 更新
   
   $$\tau_{t} = \lambda \cdot \tau_{t-1} + (1 - \lambda) \cdot \frac{1}{B}\sum_{i=1}^{B} \max(Q_u^{(i)})$$
   
   其中 $\lambda$ 是 EMA 系数，$B$ 是批次大小，$Q_u^{(i)}$ 是第 $i$ 个样本的预测概率分布。

2. **类别分布估计 ($\tilde{p}$)**：同样使用 EMA 更新
   
   $$\tilde{p}_t = \lambda \cdot \tilde{p}_{t-1} + (1 - \lambda) \cdot \frac{1}{B}\sum_{i=1}^{B} Q_u^{(i)}$$
   
   此处 $\tilde{p}$ 是一个向量，记录每个类别的平均预测概率。

3. **类别特定阈值 ($\tau_c$)**：根据类别分布动态调整各类阈值
   
   $$\tau_c = \tau \cdot \frac{\tilde{p}}{\max(\tilde{p})}$$
   
   这使得稀有类别拥有更低阈值，提高被选为伪标签的机会。

### 2.2 优势差值策略 (Δ-Margin)

传统伪标签方法可能导致累积错误。我们提出的"优势差值策略"通过两个关键步骤生成高质量伪标签：

1. **计算超阈差值**：对每个类别计算预测概率与对应阈值的差值
   
   $$\text{margin}_i = Q_u^{(i)} - \tau_c$$

2. **阈值差值筛选**：只有最大差值超过边界 $\epsilon$ 时才生成伪标签
   
   $$P_u^{(i)} = \begin{cases} 
   1, & \text{if } \max(\text{margin}_i) > \epsilon \text{ and } \arg\max(\text{margin}_i) = j \\
   0, & \text{otherwise} 
   \end{cases}$$
   
   其中 $P_u^{(i)}$ 是第 $i$ 个样本的伪标签，$j$ 是差值最大的类别索引。

### 2.3 类别平衡的伪标签选择

为了处理类别不平衡问题，我们使用按类别分组的 Top-K 选择策略：

1. 按类别分组所有伪标签候选样本
2. 从每个类别中选择置信度最高的前 K% 样本
3. 将选择的样本从无标签池中移出，添加到有标签数据集中

这确保了每个类别都能获得足够的伪标签样本，防止主流类别主导整个伪标签生成过程。

## 3. 算法流程

### 3.1 初始化阶段

```python
# 初始化自适应阈值参数
self.tau = torch.tensor(self.initial_tau, device=target_device)
self.p_tilde = torch.ones(self.num_classes, device=target_device) / self.num_classes
self.tau_c = self.tau * (self.p_tilde / torch.max(self.p_tilde))
```

### 3.2 训练流程

每个训练批次包含以下步骤：

1. **分离有标签和无标签数据**
   
   ```python
   labeled_mask = labels != -1
   unlabeled_mask = ~labeled_mask
   ```

2. **处理有标签数据**：计算监督损失
   
   ```python
   Y_l = probabilities[labeled_mask]
   L_l = labels[labeled_mask]
   current_batch_loss, sup_loss, _ = self._compute_losses(Y_l, L_l)
   ```

3. **处理无标签数据**：更新自适应阈值
   
   ```python
   Q_u = probabilities[unlabeled_mask]
   self._update_adaptive_thresholds(Q_u)
   ```

### 3.3 阈值更新机制

```python
def _update_adaptive_thresholds(self, Q_u):
    # 1. 计算每个样本的最大预测概率
    max_probs, _ = torch.max(Q_u, dim=1)
    
    # 2. 更新全局置信度阈值（EMA）
    batch_avg_confidence = torch.mean(max_probs)
    self.tau = self.lambda_ema * self.tau + (1 - self.lambda_ema) * batch_avg_confidence
    
    # 3. 更新类别平均预测概率（EMA）
    batch_class_avg = torch.mean(Q_u, dim=0)
    self.p_tilde = self.lambda_ema * self.p_tilde + (1 - self.lambda_ema) * batch_class_avg
    
    # 4. 计算每个类别的自适应阈值
    max_class_prob = torch.max(self.p_tilde)
    self.tau_c = (self.p_tilde / (max_class_prob + 1e-8)) * self.tau
    
    # 5. 限制阈值在(0, 1)之间，确保数值稳定性
    self.tau_c = torch.clamp(self.tau_c, min=1e-8, max=1.0 - 1e-8)
    self.tau = torch.clamp(self.tau, min=1e-8, max=1.0 - 1e-8)
```

### 3.4 伪标签生成

```python
def _generate_pseudo_labels(self, Q_u):
    # 1. 使用类别特定阈值检查哪些样本超过阈值
    tau_c_expanded = tau_c.unsqueeze(0)  # [1, num_classes]
    above_threshold = (Q_u > tau_c_expanded)  # [batch_size, num_classes] 布尔类型
    
    # 2. 计算每个类别的超阈差值 (预测概率 - 阈值)
    margin_values = Q_u - tau_c_expanded  # [batch_size, num_classes]
    
    # 3. 将未超过阈值的类别差值设为负无穷，确保不会被选择
    margin_values = torch.where(above_threshold, margin_values, torch.tensor(-float('inf'), device=device))
    
    # 4. 创建全零的伪标签矩阵
    P_u = torch.zeros_like(Q_u)
    
    # 5. 对于每个样本，找出最大超阈差值的类别
    for i in range(Q_u.shape[0]):
        valid_classes = torch.where(above_threshold[i])[0]
        if len(valid_classes) > 0:
            max_margin_idx = torch.argmax(margin_values[i]).item()
            max_margin_value = margin_values[i, max_margin_idx].item()
            
            # 只有当最大差值超过容忍边界时，才生成伪标签
            if max_margin_value > margin_epsilon.item():
                P_u[i, max_margin_idx] = 1.0
    
    # 6. 创建掩码：有任何一个类别被标记为1的样本
    mask = (torch.sum(P_u, dim=1) > 0).float()
    
    return P_u, mask
```

### 3.5 数据集更新

在每个周期结束时，系统会根据伪标签候选样本更新训练数据集：

```python
def _update_labeled_dataset(self, train_dataset, labeled_indices, unlabeled_indices, accelerator):
    # 按类别分组
    class_grouped_samples = {c: [] for c in range(self.num_classes)}
    
    # 遍历每个无标签样本，生成伪标签
    for sample in unlabeled_samples:
        if has_valid_pseudo_label(sample):
            pseudo_label_class = get_pseudo_label_class(sample)
            confidence = get_confidence(sample)
            class_grouped_samples[pseudo_label_class].append({
                'confidence': confidence,
                'index': sample_index
            })
    
    # 从每个类别中选择前K%的样本
    for class_id, samples in class_grouped_samples.items():
        if samples:
            sorted_samples = sorted(samples, key=lambda x: x['confidence'], reverse=True)
            num_to_select = max(1, int(len(sorted_samples) * self.top_k_percent))
            selected_samples = sorted_samples[:num_to_select]
            
            # 更新数据集
            for sample in selected_samples:
                train_dataset.data[orig_idx]['label'] = class_id
                newly_labeled_indices.append(orig_idx)
```

## 4. 与其他技术的集成

### 4.1 LoRA微调

系统灵活地集成了LoRA(Low-Rank Adaptation)技术来优化微调过程：

```python
def __init__(self, ...):
    # 如果使用LoRA，则应用LoRA配置
    if self.use_lora:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        self.model = get_peft_model(self.model, lora_config)
```

### 4.2 学习率调度器

系统实现了Warm-up + Cosine Decay学习率调度策略：

```python
def _init_lr_scheduler(self, num_training_steps):
    warmup_steps = int(num_training_steps * self.warmup_ratio)
    self.lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=self.optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
```

### 4.3 内存优化

系统实现了细致的内存管理：

```python
def _clean_memory(self, *tensors):
    for tensor in tensors:
        if tensor is not None and hasattr(tensor, 'device') and tensor.device.type == 'cuda':
            del tensor
    gc.collect()
    torch.cuda.empty_cache()
```

## 5. 实现细节与优化

### 5.1 伪标签生成的优化

通过指数移动平均(EMA)平滑地更新全局阈值和类别分布：

$$\tau_{t} = \lambda \cdot \tau_{t-1} + (1 - \lambda) \cdot \text{Avg}(\max(Q_u))$$
$$\tilde{p}_t = \lambda \cdot \tilde{p}_{t-1} + (1 - \lambda) \cdot \text{Avg}(Q_u)$$

其中，$\lambda$ 是EMA系数（典型值为0.9），$\text{Avg}$ 表示批次平均值。

### 5.2 内存优化的数据处理

在处理大量无标签数据时，方法采用批次处理和渐进式更新：

```python
# 使用小批次处理无标签数据，避免内存溢出
temp_unlabeled_loader = DataLoader(
    temp_unlabeled_dataset,
    batch_size=1,
    collate_fn=train_dataset.collate_fn,
    pin_memory=True,
    num_workers=4,
    shuffle=False
)
```

### 5.3 伪标签更新文件

将高置信度伪标签回写到标签文件，方便下次训练使用：

```python
def _update_labels_file(self, train_dataset, newly_labeled_indices, data_path, labels_file):
    # 遍历新生成的伪标签
    for idx in newly_labeled_indices:
        item = train_dataset.data[idx]
        phone_id = item['phone_id']
        pseudo_label = item['label']
        
        # 将数值标签转换为字母标签 (0->A, 1->B, 2->C, 3->D)
        label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        label_letter = label_map.get(pseudo_label)
        
        # 更新CSV文件中对应行的标签
        df.loc[df[id_col_name].astype(str) == str(phone_id), label_col_name] = label_letter
```

## 6. 方法优势与适用场景

### 6.1 解决的关键问题

1. **标注数据稀缺**：利用无标签数据生成高质量伪标签，扩充训练集
2. **类别不平衡**：自适应阈值确保稀有类别也能获得足够的样本
3. **累积误差**：Δ-Margin策略严格筛选伪标签，降低错误传播风险
4. **模型更新**：定期更新数据集并重新训练，逐步提高模型性能

### 6.2 适用场景

1. **电话营销意向分类**：电话记录标注成本高，类别分布通常不平衡
2. **医疗诊断分类**：稀有疾病样本少，但需要重点关注
3. **金融风险评估**：欺诈样本少但重要性高，需要降低阈值以提高召回率
4. **大规模音频分类**：处理大量无标签音频数据

### 6.3 参数调优建议

1. **初始阈值 (initial_tau)**：
   - 范围: 0.7-0.9
   - 较高的值（0.85+）提高伪标签质量但减少数量
   - 较低的值（0.7-0.8）增加伪标签数量但可能引入噪声

2. **EMA系数 (lambda_ema)**：
   - 范围: 0.8-0.99
   - 较高的值（0.95+）使阈值更稳定，适合大数据集
   - 较低的值（0.8-0.9）使阈值更快适应当前批次，适合小数据集

3. **Top-K百分比 (top_k_percent)**：
   - 范围: 0.05-0.2
   - 较低的值确保只选择最高质量的伪标签
   - 根据类别分布调整，稀有类别可用较高百分比

4. **容忍边界 (margin_epsilon)**：
   - 范围: 0.02-0.1
   - 较高的值（0.05+）更严格，伪标签更可靠
   - 较低的值增加伪标签数量

## 7. 效果与实验结果

在标注数据稀缺（如仅有20%标注数据）的情况下，该方法能够显著提升模型性能：

1. 通过自适应阈值机制，适应不同类别的分布特性
2. 通过优势差值策略，生成更高质量的伪标签
3. 通过类别分组的Top-K选择，确保各类别均能获得足够样本
4. 与LoRA微调相结合，提高模型参数更新效率

## 8. 总结

本文档详细介绍了一种创新的半监督自适应阈值微调方法，该方法通过动态调整置信度阈值，从无标签数据中挖掘高质量伪标签，同时确保各类别均能获得足够的伪标签样本。该方法特别适合解决标注数据稀缺和类别不平衡问题，在电话营销场景中的意向分类任务上表现出色。