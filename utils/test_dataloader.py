import os
import torch
import sys
import logging
import numpy as np
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入dataloader
from utils.dataloader import AudioEncoder, AudioFeatureExtractor, create_telemarketing_dataloader

# 设置日志级别，减少输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_batch_info(batch, batch_idx):
    """简洁打印批次的信息和标签分布"""
    if batch_idx == 0:  # 第一个批次显示完整信息
        logger.info(f"\n批次 {batch_idx}:")
        logger.info(f"批次大小: {batch['batch_size']}")
        logger.info(f"电话ID: {batch['phone_ids']}")
        logger.info(f"数值标签: {batch['labels'].tolist()}")
        
        if 'original_labels' in batch:
            logger.info(f"原始标签: {batch['original_labels']}")
            
        if 'label_onehots' in batch:
            logger.info(f"One-hot标签形状: {batch['label_onehots'].shape}")
            logger.info(f"类别数量: {batch['num_classes']}")
            logger.info(f"One-hot标签示例: {batch['label_onehots'][0]}")
            
        logger.info(f"片段数量: {batch['num_segments']}")
        
        # 打印原始特征形状
        logger.info("\n原始音频特征形状:")
        for i, features in enumerate(batch['original_features']):
            logger.info(f"  样本{i}: {features.shape}")
        
        # 打印分段特征形状
        logger.info("\n分段音频特征形状:")
        for i, segments in enumerate(batch['segment_features']):
            logger.info(f"  样本{i}: {len(segments)}个片段, 第一个片段形状: {segments[0].shape}")
    else:  # 其他批次只显示标签
        info_str = f"批次 {batch_idx} - 数值标签: {batch['labels'].tolist()}"
        if 'original_labels' in batch:
            info_str += f", 原始标签: {batch['original_labels']}"
        logger.info(info_str)

def test_standard_dataloader():
    """测试标准数据加载器"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # 创建编码器和特征提取器
    encoder = AudioEncoder.create_default_encoder()
    feature_extractor = AudioFeatureExtractor(encoder=encoder)
    
    # 创建缓存目录
    cache_dir = os.path.join(data_dir, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info("测试标准数据加载器 (不使用标签均衡)...")
    dataloader = create_telemarketing_dataloader(
        data_path=data_dir,
        feature_extractor=feature_extractor,
        encoder=encoder,
        labels_file='migrated_labels.csv',
        cache_dir=cache_dir,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # 分析前5个批次的标签分布
    logger.info("获取前5个批次数据...")
    labeled_count = 0
    total_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        print_batch_info(batch, batch_idx)
        labels = batch['labels'].tolist()
        labeled_count += sum(1 for label in labels if label != -1)  # 修改为计算不等于-1的标签数量（有标签样本）
        total_count += len(labels)
        
        if batch_idx >= 4:
            break
    
    logger.info(f"前5个批次中，有标签样本比例: {labeled_count}/{total_count} ({labeled_count/total_count:.2%})")
    
    return dataloader

def test_balanced_dataloader():
    """测试标签均衡数据加载器"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # 创建编码器和特征提取器
    encoder = AudioEncoder.create_default_encoder()
    feature_extractor = AudioFeatureExtractor(encoder=encoder)
    
    # 创建缓存目录
    cache_dir = os.path.join(data_dir, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info("\n测试标签均衡数据加载器...")
    dataloader = create_telemarketing_dataloader(
        data_path=data_dir,
        feature_extractor=feature_extractor,
        encoder=encoder,
        labels_file='migrated_labels.csv',
        cache_dir=cache_dir,
        batch_size=1,
        balance_labels=True,  # 启用标签均衡
        front_dense_ratio=0.6,
        dense_factor=2.0,
        num_workers=0
    )
    
    # 分析前5个批次的标签分布
    logger.info("获取前5个批次数据...")
    labeled_count = 0
    total_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        print_batch_info(batch, batch_idx)
        labels = batch['labels'].tolist()
        labeled_count += sum(1 for label in labels if label != -1)  # 修改为正确判断有标签样本
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
            
        labels = batch['labels'].tolist()
        labeled_count_late += sum(1 for label in labels if label != -1)
        total_count_late += len(labels)
        logger.info(f"批次 {batch_idx} - 标签: {batch['labels'].tolist()}")
    
    if total_count_late > 0:
        logger.info(f"后5个批次中，有标签样本比例: {labeled_count_late}/{total_count_late} ({labeled_count_late/total_count_late:.2%})")
    
    # 对比前后批次的标签密度
    if total_count > 0 and total_count_late > 0:
        early_density = labeled_count / total_count
        late_density = labeled_count_late / total_count_late
        logger.info(f"\n标签密度对比 - 前5个批次: {early_density:.2%}, 后5个批次: {late_density:.2%}")
        logger.info(f"密度比例: 前/后 = {early_density/late_density if late_density else 'N/A':.2f}")
    
    return dataloader

if __name__ == "__main__":
    logger.info("开始测试dataloader...")
    
    # 测试标准数据加载器
    standard_loader = test_standard_dataloader()
    
    # 测试标签均衡数据加载器
    balanced_loader = test_balanced_dataloader()
    
    logger.info("测试完成")