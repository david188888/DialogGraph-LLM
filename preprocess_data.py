#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预处理脚本：在训练前对未标注数据进行弱增强和强增强处理
"""

import os
import torch
import argparse
import logging
from utils.audio_augment import AudioAugmenter

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_unlabeled_data(
    input_path,
    output_dir, 
    weak_augment=True, 
    strong_augment=True,
    num_weak_augmentations=1,
    num_strong_augmentations=1,
    seed=42
):
    """
    预处理未标注数据，生成弱增强和强增强版本
    
    参数:
        input_path: 输入未标注数据路径
        output_dir: 输出目录
        weak_augment: 是否生成弱增强数据
        strong_augment: 是否生成强增强数据
        num_weak_augmentations: 每个样本的弱增强次数
        num_strong_augmentations: 每个样本的强增强次数
        seed: 随机种子
    """
    logger.info("开始预处理未标注数据...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化增强器
    augmenter = AudioAugmenter(seed=seed)
    
    try:
        # 加载原始未标注数据
        logger.info(f"加载未标注数据: {input_path}")
        unlabeled_data = torch.load(input_path)
        logger.info(f"成功加载数据，共 {len(unlabeled_data)} 个样本")
        
        # 处理弱增强
        if weak_augment:
            logger.info("开始生成弱增强数据...")
            weak_augmented_data = []
            
            # 保存原始数据副本
            original_copy = []
            for sample in unlabeled_data:
                sample_copy = {k: v for k, v in sample.items()}
                sample_copy["augmentation"] = "original"
                original_copy.append(sample_copy)
            
            # 应用弱增强
            for i, sample in enumerate(unlabeled_data):
                # 获取音频特征
                audio_features = sample.get("audio_features")
                if audio_features is None:
                    logger.warning(f"样本 {i} 中没有找到音频特征，跳过")
                    continue
                    
                # 进行多次弱增强
                for j in range(num_weak_augmentations):
                    augmented_sample = {k: v for k, v in sample.items()}
                    augmented_sample["audio_features"] = augmenter.weak_augment(audio_features)
                    augmented_sample["augmentation"] = "weak"
                    augmented_sample["augmentation_id"] = j
                    weak_augmented_data.append(augmented_sample)
            
            # 合并原始数据和弱增强数据
            all_weak_data = original_copy + weak_augmented_data
            
            # 保存弱增强数据
            weak_output_path = os.path.join(output_dir, "weak_augmented_data.pt")
            torch.save(all_weak_data, weak_output_path)
            logger.info(f"弱增强数据已保存到: {weak_output_path}，共 {len(all_weak_data)} 个样本")
        
        # 处理强增强
        if strong_augment:
            logger.info("开始生成强增强数据...")
            strong_augmented_data = []
            
            # 应用强增强
            for i, sample in enumerate(unlabeled_data):
                # 获取音频特征
                audio_features = sample.get("audio_features")
                if audio_features is None:
                    continue
                    
                # 进行多次强增强
                for j in range(num_strong_augmentations):
                    augmented_sample = {k: v for k, v in sample.items()}
                    augmented_sample["audio_features"] = augmenter.strong_augment(audio_features)
                    augmented_sample["augmentation"] = "strong"
                    augmented_sample["augmentation_id"] = j
                    strong_augmented_data.append(augmented_sample)
            
            # 保存强增强数据
            strong_output_path = os.path.join(output_dir, "strong_augmented_data.pt")
            torch.save(strong_augmented_data, strong_output_path)
            logger.info(f"强增强数据已保存到: {strong_output_path}，共 {len(strong_augmented_data)} 个样本")
        
        # 保存配对数据（弱增强+强增强）用于FixMatch
        if weak_augment and strong_augment:
            logger.info("生成弱增强-强增强配对数据...")
            paired_data = []
            
            for i, sample in enumerate(unlabeled_data):
                audio_features = sample.get("audio_features")
                if audio_features is None:
                    continue
                
                # 每个原始样本生成一对弱增强和强增强
                weak_features = augmenter.weak_augment(audio_features)
                strong_features = augmenter.strong_augment(audio_features)
                
                paired_sample = {k: v for k, v in sample.items()}
                paired_sample["original_features"] = audio_features
                paired_sample["weak_audio_features"] = weak_features
                paired_sample["strong_audio_features"] = strong_features
                paired_sample["is_labeled"] = False
                paired_data.append(paired_sample)
            
            # 保存配对数据
            paired_output_path = os.path.join(output_dir, "fixmatch_paired_data.pt")
            torch.save(paired_data, paired_output_path)
            logger.info(f"FixMatch配对数据已保存到: {paired_output_path}，共 {len(paired_data)} 个样本")
        
        logger.info("数据预处理完成！")
        return True
        
    except Exception as e:
        logger.error(f"预处理数据失败: {e}")
        return False

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='未标注数据预处理工具')
    parser.add_argument('--input', type=str, required=True, help='输入未标注数据路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--weak-augment', action='store_true', default=True, help='是否生成弱增强数据')
    parser.add_argument('--strong-augment', action='store_true', default=True, help='是否生成强增强数据')
    parser.add_argument('--num-weak', type=int, default=1, help='每个样本的弱增强次数')
    parser.add_argument('--num-strong', type=int, default=1, help='每个样本的强增强次数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 执行预处理
    preprocess_unlabeled_data(
        args.input,
        args.output_dir,
        args.weak_augment,
        args.strong_augment,
        args.num_weak,
        args.num_strong,
        args.seed
    ) 