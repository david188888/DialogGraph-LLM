import torch
import numpy as np
import os
import logging
from tqdm import tqdm
import random

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioAugmenter:
    """音频特征增强器"""
    
    def __init__(self, seed=42):
        """
        初始化音频增强器
        
        参数:
            seed: 随机种子，用于保证增强结果的可复现性
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def weak_augment(self, audio_features):
        """
        弱数据增强，适用于音频特征的简单增强
        
        参数:
            audio_features: 音频特征张量
            
        返回:
            增强后的音频特征
        """
        # 使用轻微的高斯噪声
        noise = 0.1 * torch.randn_like(audio_features)
        return audio_features + noise
    
    def strong_augment(self, audio_features):
        """
        强数据增强，更复杂的增强方法
        
        参数:
            audio_features: 音频特征张量
            
        返回:
            增强后的音频特征
        """
        # 添加较强的高斯噪声
        noise = 0.2 * torch.randn_like(audio_features)
        # 随机缩放
        scale = 1.0 + 0.1 * torch.randn_like(audio_features)
        
        return (audio_features + noise) * scale
    
    def time_mask(self, audio_features, mask_param=10, num_masks=2):
        """
        时间掩码增强，类似于SpecAugment中的time masking
        
        参数:
            audio_features: 音频特征张量 [length, dim]
            mask_param: 最大掩码长度
            num_masks: 掩码数量
            
        返回:
            增强后的音频特征
        """
        augmented = audio_features.clone()
        length = audio_features.shape[0]
        
        for i in range(num_masks):
            mask_length = random.randint(1, mask_param)
            mask_start = random.randint(0, length - mask_length)
            augmented[mask_start:mask_start + mask_length, :] = 0
            
        return augmented
    
    def feature_mask(self, audio_features, mask_param=10, num_masks=2):
        """
        特征掩码增强，类似于SpecAugment中的frequency masking
        
        参数:
            audio_features: 音频特征张量 [length, dim]
            mask_param: 最大掩码长度
            num_masks: 掩码数量
            
        返回:
            增强后的音频特征
        """
        augmented = audio_features.clone()
        dim = audio_features.shape[1]
        
        for i in range(num_masks):
            mask_length = random.randint(1, mask_param)
            mask_start = random.randint(0, dim - mask_length)
            augmented[:, mask_start:mask_start + mask_length] = 0
            
        return augmented
    
    def pitch_shift(self, audio_features, shift_range=0.2):
        """
        模拟音高变化的增强
        
        参数:
            audio_features: 音频特征张量
            shift_range: 变化范围
            
        返回:
            增强后的音频特征
        """
        # 在特征空间进行简单的变换模拟音高变化
        shift = random.uniform(-shift_range, shift_range)
        return audio_features * (1.0 + shift)
    
    def augment_dataset(self, data_path, output_path, augment_type='weak', num_augmentations=1):
        """
        增强整个数据集
        
        参数:
            data_path: 原始数据路径
            output_path: 增强数据输出路径
            augment_type: 增强类型，'weak'或'strong'
            num_augmentations: 每个样本的增强次数
            
        返回:
            增强后的数据集
        """
        logger.info(f"开始对数据集 {data_path} 进行{augment_type}增强...")
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 加载数据集
        # 注意：这里的数据加载方式需要根据实际数据格式进行调整
        try:
            # 假设数据以torch格式保存
            original_data = torch.load(data_path)
            logger.info(f"成功加载数据集，共 {len(original_data)} 个样本")
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            return None
        
        augmented_data = []
        for i, sample in enumerate(tqdm(original_data, desc=f"正在进行{augment_type}增强")):
            # 获取音频特征
            audio_features = sample.get("audio_features")
            
            if audio_features is None:
                logger.warning(f"样本 {i} 中没有找到音频特征，跳过")
                continue
                
            # 保留原始样本
            augmented_data.append(sample)
            
            # 进行多次增强
            for j in range(num_augmentations):
                augmented_sample = sample.copy()
                
                if augment_type == 'weak':
                    augmented_sample["audio_features"] = self.weak_augment(audio_features)
                    augmented_sample["augmentation"] = "weak"
                else:  # strong
                    augmented_sample["audio_features"] = self.strong_augment(audio_features)
                    augmented_sample["augmentation"] = "strong"
                
                augmented_sample["augmentation_id"] = j
                augmented_data.append(augmented_sample)
        
        # 保存增强后的数据
        output_file = os.path.join(output_path, f"{augment_type}_augmented_data.pt")
        torch.save(augmented_data, output_file)
        
        logger.info(f"数据增强完成，共生成 {len(augmented_data)} 个样本")
        logger.info(f"增强数据已保存到: {output_file}")
        
        return augmented_data
    
    def preprocess_unlabeled_data(self, unlabeled_data_path, output_path):
        """
        预处理无标签数据，进行弱增强
        
        参数:
            unlabeled_data_path: 无标签数据路径
            output_path: 处理后数据输出路径
            
        返回:
            处理后的无标签数据
        """
        return self.augment_dataset(unlabeled_data_path, output_path, augment_type='weak', num_augmentations=1)


# 如果直接运行这个脚本，则执行下面的代码
if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='音频数据增强工具')
    parser.add_argument('--data-path', type=str, required=True, help='原始数据路径')
    parser.add_argument('--output-path', type=str, required=True, help='输出路径')
    parser.add_argument('--augment-type', type=str, default='weak', choices=['weak', 'strong'], help='增强类型')
    parser.add_argument('--num-augmentations', type=int, default=1, help='每个样本的增强次数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 创建增强器并处理数据
    augmenter = AudioAugmenter(seed=args.seed)
    augmenter.augment_dataset(
        args.data_path, 
        args.output_path, 
        augment_type=args.augment_type, 
        num_augmentations=args.num_augmentations
    ) 