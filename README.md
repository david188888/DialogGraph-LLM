# AudioLLM-Telemarketing
Use LLM combined with defined rule systems and abductive learning to analyze users' purchasing intentions in the audio field of telemarketing

# 半监督自适应阈值微调训练系统

本项目实现了一个基于FixMatch算法的半监督学习系统，并引入了创新的类别自适应阈值策略，用于电话营销场景中的意向分类任务。

## 系统架构

### 核心组件

1. **数据处理模块** (`AudioDialogueDataset`)
   - 支持有标签和无标签数据的混合训练
   - 实现数据增强策略（弱增强和强增强）
   - 动态批次生成，确保有标签数据均匀分布

2. **训练器模块** (`AdaptiveThresholdTrainer`)
   - 集成LoRA微调策略
   - 实现类别自适应阈值机制
   - 支持半监督学习的FixMatch算法

3. **图网络模块** (`DialogueGraphModel`)
   - 处理对话结构信息
   - 生成图嵌入表示

### 创新特点

- 类别自适应阈值策略
- 动态损失权重调整
- EMA（指数移动平均）更新机制
- 分层的置信度评估系统

## 工作流程

### 1. 数据预处理阶段

```python
# 数据加载和预处理
train_dataset = AudioDialogueDataset(
    data_path=train_data_path,
    tokenizer=tokenizer,
    labeled_ratio=0.1  # 10%有标签数据
)
```

- 加载原始数据
- 划分有标签和无标签数据
- 生成数据批次分组
- 应用数据增强转换

### 2. 模型初始化阶段

```python
trainer = AdaptiveThresholdTrainer(
    model_name=model_name,
    graph_config=graph_config,
    output_dir=output_dir,
    threshold_init=0.7,
    num_classes=4
)
```

- 初始化预训练语言模型
- 配置LoRA参数
- 初始化图网络模型
- 设置自适应阈值参数

### 3. 训练循环

#### 3.1 有标签数据处理
- 计算监督损失
- 更新模型参数

#### 3.2 无标签数据处理
1. 生成置信度分布：
   ```python
   confidence_probs = self._get_confidence_distribution(
       confidence_outputs, 
       confidence_input_ids[unlabeled_mask]
   )
   ```

2. 更新自适应阈值：
   ```python
   class_thresholds = self._update_thresholds(confidence_probs)
   ```

3. 生成伪标签：
   ```python
   pseudo_labels, confidence_mask = self._generate_pseudo_labels(
       confidence_probs, 
       class_thresholds
   )
   ```

4. 计算无标签损失

### 4. 监控和评估

- 实时监控训练指标
- 定期评估模型性能
- 记录详细的训练日志
- 支持wandb可视化

## 实现细节

### 1. 类别自适应阈值策略

```python
def _update_thresholds(self, confidence_probs):
    # 1. 计算每个样本的最大预测概率
    max_probs, pred_classes = torch.max(confidence_probs, dim=1)
    
    # 2. 更新全局阈值
    mean_confidence = max_probs.mean().item()
    self.threshold = self.threshold_momentum * self.threshold + \
                    (1 - self.threshold_momentum) * mean_confidence
    
    # 3. 更新类别概率
    # ...
    
    # 4. 计算类别自适应阈值
    class_thresholds = normalized_class_probs * self.threshold * self.num_classes
    
    return class_thresholds
```

### 2. 置信度评估系统

```python
def _parse_confidence_output(self, output_text):
    # 解析模型输出的置信度分布
    confidence_pattern = r"置信度分布:\s*(高意向:[\d.]+,中意向:[\d.]+,低意向:[\d.]+,无意向:[\d.]+)"
    # ...
```

### 3. 损失计算

- 有标签损失：交叉熵损失
- 无标签损失：基于伪标签的交叉熵损失
- 动态权重调整

## 使用指南

### 1. 环境配置

```bash
pip install -r requirements.txt
```

### 2. 数据准备

- 准备训练数据集
- 设置标签比例
- 配置数据路径

### 3. 训练启动

```python
python train.py
```

### 4. 参数配置

主要参数说明：
- `labeled_ratio`: 有标签数据比例
- `threshold_init`: 初始阈值
- `class_threshold_momentum`: 类别阈值动量
- `lambda_u_max`: 无标签损失最大权重

## 监控和调试

### 1. 训练日志

```python
logger.info(f"Epoch {epoch+1}/{num_epochs} completed:")
logger.info(f"  Average loss: {epoch_avg_loss:.4f}")
logger.info(f"  Labeled loss: {labeled_avg_loss:.4f}")
# ...
```

### 2. Wandb集成

```python
if use_wandb:
    wandb.log({
        "loss": epoch_loss / (step + 1),
        "labeled_loss": labeled_loss / max(num_labeled, 1),
        # ...
    })
```

## 注意事项

1. 数据预处理
   - 确保数据格式正确
   - 注意数据增强参数设置

2. 训练过程
   - 监控阈值变化
   - 关注伪标签质量

3. 模型保存
   - 定期保存检查点
   - 备份最佳模型

## 常见问题

1. 内存管理
   - 适当调整批次大小
   - 使用梯度累积

2. 训练稳定性
   - 调整学习率
   - 优化阈值更新策略

3. 性能优化
   - 使用混合精度训练
   - 优化数据加载流程
 