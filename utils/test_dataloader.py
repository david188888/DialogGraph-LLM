import os
import torch
import sys
import logging
import numpy as np
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入需要的类
from utils.dataloader import DataLoaderConfig, AudioSegmentDataset, LabelBalancedSampler
from utils.dataloader import DEFAULT_AUDIO_DIR, DEFAULT_SEGMENTS_DIR, DEFAULT_MODEL_PATH

# 设置日志级别，减少输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_batch_info(batch, batch_idx):
    """简洁打印批次的信息和标签分布"""
    if batch_idx == 0:  # 第一个批次显示完整信息
        logger.info(f"电话ID: {batch['phone_ids']}")
        logger.info(f"数值标签: {batch['label']}")
            
        logger.info(f"片段数量: {batch['num_segments']}")
        
        # 打印分段特征形状
        logger.info("\n分段音频特征形状:")

    else:  # 其他批次只显示标签
        info_str = f"批次 {batch_idx} - 数值标签: {batch['label']}"
        logger.info(info_str)

def test_standard_dataloader():
    """测试标准数据加载器"""
    # 使用自定义路径
    data_dir = "/data/shared/Qwen/data"
    
    # 创建缓存目录
    cache_dir = os.path.join(data_dir, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info("测试标准数据加载器 (不使用标签均衡)...")

    # 1. 创建配置
    config = DataLoaderConfig(
        data_path=data_dir,
        labels_file='migrated_labels.csv',
        cache_dir=cache_dir,
        batch_size=1, # 测试时常用 batch_size=1
        shuffle=True,
        num_workers=0,
        balance_labels=False, # 标准加载器不使用均衡
        model_path=DEFAULT_MODEL_PATH  # 使用默认模型路径
    )

    # 2. 创建数据集
    dataset = AudioSegmentDataset(
        data_path=config.data_path,
        model_path=DEFAULT_MODEL_PATH,
        labels_file=config.labels_file,
        cache_dir=config.cache_dir,
        audio_dir=DEFAULT_AUDIO_DIR,
        segments_dir=DEFAULT_SEGMENTS_DIR
    )

    # 3. 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn
    )
    
    # 分析前5个批次的标签分布
    logger.info("获取前5个批次数据...")
    labeled_count = 0
    total_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        print_batch_info(batch, batch_idx)
        labels = batch['labels'].tolist()
        labeled_count += sum(1 for label in labels if label is None)  # 修改为计算不等于-1的标签数量（有标签样本）
        total_count += len(labels)
        
        if batch_idx >= 4:
            break
    
    logger.info(f"前5个批次中，有标签样本比例: {labeled_count}/{total_count} ({labeled_count/total_count:.2%})")
    
    return dataloader

def test_balanced_dataloader():
    """测试标签均衡数据加载器"""
    # 使用自定义路径
    data_dir = "/data/shared/Qwen/data"
    
    # 创建缓存目录
    cache_dir = os.path.join(data_dir, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info("\n测试标签均衡数据加载器...")

    # 1. 创建配置
    config = DataLoaderConfig(
        data_path=data_dir,
        labels_file='migrated_labels.csv',
        cache_dir=cache_dir,
        batch_size=1,
        shuffle=False, # 使用 sampler 时 shuffle 必须为 False
        num_workers=0,
        balance_labels=True, # 启用标签均衡
        front_dense_ratio=0.6,
        dense_factor=2.0,
        model_path=DEFAULT_MODEL_PATH
    )

    # 2. 创建数据集
    dataset = AudioSegmentDataset(
        data_path=config.data_path,
        model_path=DEFAULT_MODEL_PATH,
        labels_file=config.labels_file,
        cache_dir=config.cache_dir,
        audio_dir=DEFAULT_AUDIO_DIR,
        segments_dir=DEFAULT_SEGMENTS_DIR
    )

    # 3. 创建 Sampler
    sampler = LabelBalancedSampler(
        dataset=dataset,
        batch_size=config.batch_size,
        front_dense_ratio=config.front_dense_ratio,
        dense_factor=config.dense_factor
    )
    print(f"标签均衡采样器已创建。")

    # 4. 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False, # sampler 和 shuffle 互斥
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn
    )
    
    # 分析前5个批次的标签分布
    logger.info("获取前5个批次数据...")
    labeled_count = 0
    total_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        print_batch_info(batch, batch_idx)
        labels = batch['label']
        labeled_count += sum(1 for label in labels if label is None)  # 修改为正确判断有标签样本
        total_count += len(labels)
        
        if batch_idx >= 4:
            break
    
    logger.info(f"前5个批次中，有标签样本比例: {labeled_count}/{total_count} ({labeled_count/total_count:.2%})")
    
    # 分析后5个批次的标签分布
    logger.info("跳过中间批次，获取后5个批次数据...")
    labeled_count_late = 0
    total_count_late = 0
    
    # 跳过中间批次
    batch_count = len(dataloader)
    skip_to = max(5, batch_count - 5)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < skip_to:
            continue
        if batch_idx >= batch_count:
            break
            
        labels = batch['label']
        print(label for label in labels)
        labeled_count_late += sum(1 for label in labels if label is None)
        total_count_late += len(labels)
        logger.info(f"批次 {batch_idx} - 标签: {batch['label']}")
    
    if total_count_late > 0:
        logger.info(f"后5个批次中，有标签样本比例: {labeled_count_late}/{total_count_late} ({labeled_count_late/total_count_late:.2%})")
    
    # 对比前后批次的标签密度
    if total_count > 0 and total_count_late > 0:
        early_density = labeled_count / total_count
        late_density = labeled_count_late / total_count_late
        logger.info(f"\n标签密度对比 - 前5个批次: {early_density:.2%}, 后5个批次: {late_density:.2%}")
        
        # 避免除零错误的正确方式
        if late_density > 0:
            ratio = early_density / late_density
            logger.info(f"密度比例: 前/后 = {ratio:.2f}")
        else:
            logger.info(f"密度比例: 前/后 = 无法计算 (后5个批次无有标签样本)")
    
    return dataloader

if __name__ == "__main__":
    logger.info("开始测试dataloader...")
    
    # 测试标准数据加载器
    # standard_loader = test_standard_dataloader()
    
    # 测试标签均衡数据加载器
    balanced_loader = test_balanced_dataloader()
    
    logger.info("测试完成")