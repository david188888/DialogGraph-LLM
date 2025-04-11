#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import shutil
import random
import argparse
from pathlib import Path
import logging

"""
数据集分割脚本
将原始数据分割为训练集、验证集和测试集
特殊要求：测试集必须全部是有标签的数据，且只包含50个音频样本
"""

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetSplitter:
    """数据集分割器"""
    
    def __init__(self, 
                data_dir, 
                labels_file='migrated_labels.csv',
                output_dir=None, 
                test_size=50, 
                val_ratio=0.15, 
                copy_files=True,
                random_seed=42):
        """
        初始化数据集分割器
        
        参数:
            data_dir: 数据根目录路径
            labels_file: 标签文件名
            output_dir: 输出目录，默认为data_dir
            test_size: 测试集大小 (固定为50个样本)
            val_ratio: 验证集比例 (占剩余数据的比例)
            copy_files: 是否复制文件 (True)或移动文件 (False)
            random_seed: 随机种子，确保结果可重现
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        self.labels_file = self.data_dir / labels_file
        self.audio_dir = self.data_dir / 'audio'
        self.segments_dir = self.data_dir / 'segments'
        self.test_size = test_size
        self.val_ratio = val_ratio
        self.copy_files = copy_files
        self.random_seed = random_seed
        
        # 设置随机种子
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # 标签数据集
        self.labels_df = None
        
        # 记录所有带标签的ID
        self.labeled_ids = []
        # 记录所有文件ID (不带标签的)
        self.all_audio_ids = []
        
        # 数据集划分结果
        self.test_ids = []
        self.val_ids = []
        self.train_ids = []
        
    def load_data(self):
        """加载数据和标签"""
        logger.info(f"正在加载标签文件: {self.labels_file}")
        
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"标签文件不存在: {self.labels_file}")
        
        try:
            self.labels_df = pd.read_csv(self.labels_file)
            logger.info(f"成功加载标签文件，共 {len(self.labels_df)} 条记录")
        except Exception as e:
            logger.error(f"加载标签文件失败: {e}")
            raise
        
        # 找出所有带标签的ID
        labeled_samples = self.labels_df[self.labels_df['label'].notna() & 
                                        (self.labels_df['label'] != "") & 
                                        (self.labels_df['label'] != "none")]
        
        # 提取唯一的音频ID (标签文件中可能包含segments_id，我们需要原始音频ID)
        audio_ids_from_labels = []
        for audio_id in labeled_samples['audio_id']:
            # 如果ID包含下划线，说明是segment_id，我们取下划线前的部分作为audio_id
            if '_' in audio_id:
                audio_id = audio_id.split('_')[0]
            audio_ids_from_labels.append(audio_id)
        
        # 获取唯一的音频ID
        self.labeled_ids = list(set(audio_ids_from_labels))
        logger.info(f"找到 {len(self.labeled_ids)} 个带标签的音频ID")
        
        # 读取音频目录，获取所有音频文件
        audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
        self.all_audio_ids = [f.split('.')[0] for f in audio_files]
        logger.info(f"音频目录中共有 {len(self.all_audio_ids)} 个音频文件")
        
        # 检查是否有足够的带标签样本用于测试集
        if len(self.labeled_ids) < self.test_size:
            logger.warning(f"带标签的样本数量 ({len(self.labeled_ids)}) 小于测试集大小 ({self.test_size})!")
            logger.warning(f"将使用所有带标签样本作为测试集")
            self.test_size = len(self.labeled_ids)
    
    def split_dataset(self):
        """划分数据集"""
        logger.info("开始划分数据集...")
        
        # 确保标签已加载
        if self.labeled_ids is None or len(self.labeled_ids) == 0:
            self.load_data()
        
        # 随机打乱带标签的ID
        random.shuffle(self.labeled_ids)
        
        # 选取测试集 (全部带标签)
        self.test_ids = self.labeled_ids[:self.test_size]
        
        # 剩余带标签的ID和无标签的ID
        remaining_labeled_ids = self.labeled_ids[self.test_size:]
        unlabeled_ids = [audio_id for audio_id in self.all_audio_ids if audio_id not in self.labeled_ids]
        
        # 合并剩余的ID
        remaining_ids = remaining_labeled_ids + unlabeled_ids
        random.shuffle(remaining_ids)
        
        # 计算验证集大小
        val_size = int(len(remaining_ids) * self.val_ratio)
        
        # 分配验证集和训练集
        self.val_ids = remaining_ids[:val_size]
        self.train_ids = remaining_ids[val_size:]
        
        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {len(self.train_ids)} 个样本")
        logger.info(f"  验证集: {len(self.val_ids)} 个样本")
        logger.info(f"  测试集: {len(self.test_ids)} 个样本 (全部带标签)")
    
    def create_dataset_structure(self):
        """创建数据集目录结构"""
        logger.info("创建数据集目录结构...")
        
        # 创建数据集目录
        for split in ['train', 'val', 'test']:
            for subdir in ['audio', 'segments']:
                dir_path = self.output_dir / split / subdir
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"创建目录: {dir_path}")
        
        # 创建标签文件
        self._create_label_files()
    
    def _copy_or_move_files(self, src_file, dst_file):
        """复制或移动文件"""
        if self.copy_files:
            shutil.copy2(src_file, dst_file)
        else:
            shutil.move(src_file, dst_file)
    
    def _create_label_files(self):
        """为各数据集创建标签文件"""
        logger.info("创建标签文件...")
        
        # 为每个数据集创建标签文件
        for split, ids in [('train', self.train_ids), ('val', self.val_ids), ('test', self.test_ids)]:
            # 筛选属于该数据集的标签
            split_labels = self.labels_df[self.labels_df['audio_id'].apply(
                lambda x: x.split('_')[0] if '_' in x else x).isin(ids)]
            
            # 保存标签文件
            label_file = self.output_dir / split / 'migrated_labels.csv'
            split_labels.to_csv(label_file, index=False)
            logger.info(f"创建标签文件: {label_file}, 包含 {len(split_labels)} 条记录")
    
    def distribute_files(self):
        """分发文件到各数据集目录"""
        logger.info("开始分发文件到各数据集目录...")
        
        # 处理音频文件
        self._distribute_audio_files()
        
        # 处理片段文件
        self._distribute_segment_files()
        
        logger.info("文件分发完成")
    
    def _distribute_audio_files(self):
        """分发音频文件"""
        logger.info("分发音频文件...")
        
        # 为每个数据集分发音频文件
        for split, ids in [('train', self.train_ids), ('val', self.val_ids), ('test', self.test_ids)]:
            count = 0
            for audio_id in ids:
                src_file = self.audio_dir / f"{audio_id}.wav"
                if not os.path.exists(src_file):
                    logger.warning(f"找不到音频文件: {src_file}")
                    continue
                
                dst_file = self.output_dir / split / 'audio' / f"{audio_id}.wav"
                self._copy_or_move_files(src_file, dst_file)
                count += 1
            
            logger.info(f"为 {split} 集分发了 {count} 个音频文件")
    
    def _distribute_segment_files(self):
        """分发片段文件"""
        logger.info("分发片段文件...")
        
        # 获取所有片段文件
        segment_files = [f for f in os.listdir(self.segments_dir) if f.endswith('.wav')]
        
        # 为每个数据集分发片段文件
        for split, ids in [('train', self.train_ids), ('val', self.val_ids), ('test', self.test_ids)]:
            count = 0
            for segment_file in segment_files:
                # 从文件名提取音频ID
                if '_' in segment_file:
                    audio_id = segment_file.split('_')[0]
                else:
                    audio_id = segment_file.split('.')[0]
                
                # 检查该音频ID是否属于当前数据集
                if audio_id in ids:
                    src_file = self.segments_dir / segment_file
                    dst_file = self.output_dir / split / 'segments' / segment_file
                    self._copy_or_move_files(src_file, dst_file)
                    count += 1
            
            logger.info(f"为 {split} 集分发了 {count} 个片段文件")
    
    def run(self):
        """运行数据集分割流程"""
        logger.info("开始数据集分割流程...")
        
        # 加载数据
        self.load_data()
        
        # 划分数据集
        self.split_dataset()
        
        # 创建数据集目录结构
        self.create_dataset_structure()
        
        # 分发文件
        self.distribute_files()
        
        logger.info("数据集分割完成!")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据集分割工具')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据根目录 (默认: ./data)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: 与data-dir相同)')
    parser.add_argument('--labels-file', type=str, default='migrated_labels.csv',
                        help='标签文件名 (默认: migrated_labels.csv)')
    parser.add_argument('--test-size', type=int, default=50,
                        help='测试集大小 (默认: 50)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='验证集比例 (默认: 0.15)')
    parser.add_argument('--copy', action='store_true',
                        help='复制文件而非移动 (默认: 复制)')
    parser.add_argument('--move', action='store_true',
                        help='移动文件而非复制')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 处理copy和move的冲突
    if args.move:
        args.copy = False
    else:
        args.copy = True
    
    return args

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建数据集分割器
    splitter = DatasetSplitter(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        copy_files=args.copy,
        random_seed=args.seed
    )
    
    # 运行分割流程
    splitter.run()

if __name__ == "__main__":
    main() 