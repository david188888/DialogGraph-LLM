import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import pandas as pd
import librosa
import soundfile as sf
import tqdm
import logging
import torch.nn as nn
from utils.model import AudioEncoder, AudioFeatureExtractor

class AudioSegmentDataset(Dataset):
    
    def __init__(self, data_path, feature_extractor=None, encoder=None, labels_file='migrated_labels.csv',
                 max_segments=10, cache_dir='features_cache', system_prompt=None):
        """
        初始化音频分段数据集
        
        参数:
            data_path: 数据根目录路径
            feature_extractor: 音频特征提取器实例，如果None则创建默认提取器
            encoder: 音频编码器，用于进一步处理特征
            labels_file: 标签文件名称
            max_segments: 每个对话最多使用的片段数量
            cache_dir: 特征缓存目录
            system_prompt: 系统提示文本
        """
        self.data_path = data_path
        self.segments_dir = os.path.join(data_path, 'segments')
        self.audio_dir = os.path.join(data_path, 'audio')
        self.labels_file = os.path.join(data_path, labels_file)
        self.max_segments = max_segments
        self.cache_dir = os.path.join(data_path, cache_dir) if cache_dir else None
        self.system_prompt = system_prompt
        
        # 创建特征提取器
        if feature_extractor is None:
            # 如果没有提供特征提取器，则创建默认特征提取器
            if encoder is None:
                # 如果没有提供编码器，则创建默认编码器
                encoder = AudioEncoder.create_default_encoder()
            self.feature_extractor = AudioFeatureExtractor(encoder=encoder)
        else:
            self.feature_extractor = feature_extractor
        
        # 标签映射字典，用于将字母标签映射为数字和one-hot
        self.label_to_id = {}
        self.id_to_label = {}
        self.num_classes = 0
        
        # 加载数据
        self._load_data()
        
        # 构建标签映射
        self._build_label_mapping()
    
    def _build_label_mapping(self):
        """构建标签到ID的映射及确定类别数量"""
        # 明确指定有效标签为A、B、C、D
        valid_labels = ['A', 'B', 'C', 'D']
        self.num_classes = len(valid_labels)
        
        # 为每个标签分配ID
        for i, label in enumerate(valid_labels):
            self.label_to_id[label] = i
            self.id_to_label[i] = label
        
        print(f"标签映射: {self.label_to_id}")
        print(f"总共 {self.num_classes} 种类别")
    
    def _label_to_onehot(self, label):
        """
        将标签转换为one-hot向量，空标签返回全零向量
        
        参数:
            label: 标签值（可以是字符串或数字）
            
        返回:
            one-hot张量，形状为 [num_classes]
        """
        # 明确检查空标签或none标签情况
        if label is None or label == '' or label == -1 or str(label).lower() == 'none':
            return torch.zeros(self.num_classes)
        
        # 将标签转换为字符串并获取标签ID
        str_label = str(label)
        if str_label in self.label_to_id:
            label_id = self.label_to_id[str_label]
            
            # 创建one-hot向量
            one_hot = torch.zeros(self.num_classes)
            one_hot[label_id] = 1.0
            return one_hot
        else:
            # 未知标签，返回全零向量
            print(f"警告：未识别的标签值 '{str_label}'，返回全零向量")
            return torch.zeros(self.num_classes)
    
    def _load_data(self):
        """加载数据，建立音频片段与标签的对应关系，不限制片段数量"""
        self.data = []
        
        try:
            # 加载标签数据
            if os.path.exists(self.labels_file):
                # 假设CSV格式为: 音频ID,标签1(none或空),标签2(ABCD)
                labels_df = pd.read_csv(self.labels_file, header=None)
                
                # 将DataFrame转换为字典，使用第一列作为键
                labels_dict = {}
                for _, row in labels_df.iterrows():
                    if len(row) >= 3:  # 确保至少有三列
                        audio_id = str(row[0])
                        label_value = row[2] if pd.notna(row[2]) and row[2] != 'none' else None
                        labels_dict[audio_id] = {'label': label_value, 'speaker': None}
                        
            else:
                labels_dict = {}
                print(f"警告：找不到标签文件 {self.labels_file}")
            
            # 扫描segments目录，获取所有切分好的音频片段
            if not os.path.exists(self.segments_dir):
                raise FileNotFoundError(f"找不到切分音频目录: {self.segments_dir}")
            
            # 获取所有原始电话ID，直接从audio目录获取
            phone_ids = set()
            for file_name in os.listdir(self.audio_dir):
                if file_name.endswith('.wav'):
                    phone_id = file_name.split('.')[0]  # 去掉扩展名
                    phone_ids.add(phone_id)
            
            # 为每个电话收集所有切分片段
            for phone_id in phone_ids:
                # 检查原始音频文件是否存在
                original_audio_file = os.path.join(self.audio_dir, f"{phone_id}.wav")
                if not os.path.exists(original_audio_file):
                    print(f"警告：找不到原始音频文件: {original_audio_file}")
                    continue
                
                # 收集片段文件 - 使用更准确的匹配方式
                segment_files = []
                for file_name in os.listdir(self.segments_dir):
                    # 使用startswith而不是split，这样可以处理电话ID本身包含下划线的情况
                    if file_name.startswith(f"{phone_id}_") and file_name.endswith('.wav'):
                        segment_files.append(os.path.join(self.segments_dir, file_name))
                
                # 如果找不到片段，跳过该电话
                if not segment_files:
                    print(f"警告：电话ID {phone_id} 没有找到对应的片段文件")
                    continue
                
                # 根据片段编号排序
                try:
                    segment_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
                except Exception as e:
                    print(f"对片段文件排序时出错: {e}，使用文件名排序")
                    segment_files.sort()  # 退回到简单的文件名排序
                
                # 获取标签信息
                label_info = labels_dict.get(phone_id, {})
                label = label_info.get('label', None)
                speaker = label_info.get('speaker', None)
                
                # 添加数据项
                self.data.append({
                    'phone_id': phone_id,
                    'original_audio_file': original_audio_file,
                    'segment_files': segment_files,
                    'label': label,
                    'speaker': speaker,
                    'num_segments': len(segment_files)
                })
            
            total_segments = sum(item['num_segments'] for item in self.data)
            avg_segments = total_segments / len(self.data) if self.data else 0
            max_segments = max(item['num_segments'] for item in self.data) if self.data else 0
            print(f"加载了 {len(self.data)} 个电话对话，总共 {total_segments} 个片段")
            print(f"平均每个对话 {avg_segments:.2f} 个片段，最多 {max_segments} 个片段")
            print("注意：因为保留了所有片段，建议使用batch_size=1以避免内存问题")
            
            # 统计各类标签数量
            label_counts = {}
            for item in self.data:
                label = item.get('label')
                if label is not None:
                    label = str(label)
                    label_counts[label] = label_counts.get(label, 0) + 1
                else:
                    label_counts['none'] = label_counts.get('none', 0) + 1
            
            print(f"标签分布: {label_counts}")
            
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.data = []
    
    def __len__(self):
        """返回数据集中样本数量"""
        return len(self.data)
    
    def _get_audio_segments(self, item):
        """
        获取音频片段的特征和原始音频的特征，使用两个总的pt文件分别存储
        
        参数:
            item: 数据项，包含原始音频和片段音频的路径
            
        返回:
            原始音频特征和分段特征列表的元组
        """
        phone_id = item['phone_id']
        original_audio_file = item['original_audio_file']
        segment_files = item['segment_files']
        
        # 定义两个总的缓存文件路径
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            original_features_cache = os.path.join(self.cache_dir, "original_features.pt")
            segment_features_cache = os.path.join(self.cache_dir, "segment_features.pt")
        else:
            original_features_cache = None
            segment_features_cache = None
        
        # --- 处理原始音频特征 ---
        original_features = None
        
        # 尝试从总缓存加载原始音频特征
        if original_features_cache and os.path.exists(original_features_cache):
            try:
                features_dict = torch.load(original_features_cache)
                # 检查当前phone_id是否在缓存中
                if phone_id in features_dict:
                    original_features = features_dict[phone_id]['features']
            except Exception as e:
                print(f"加载原始音频特征缓存失败: {str(e)}")
        else:
            # 如果缓存文件不存在，创建一个空字典
            features_dict = {}
        
        # 如果缓存中没有当前phone_id的特征，则提取特征
        if original_features is None:
            original_features = self.feature_extractor.extract_features(original_audio_file)
            
            # 将新提取的特征添加到字典中
            if original_features_cache:
                features_dict[phone_id] = {
                    'phone_id': phone_id,
                    'audio_path': original_audio_file,
                    'features': original_features,
                    'extraction_time': torch.tensor(pd.Timestamp.now().timestamp())
                }
                # 保存更新后的字典
                torch.save(features_dict, original_features_cache)
        
        # --- 处理分段音频特征 ---
        segment_features = []
        
        # 尝试从总缓存加载分段音频特征
        segment_dict = {}
        if segment_features_cache and os.path.exists(segment_features_cache):
            try:
                segment_dict = torch.load(segment_features_cache)
            except Exception as e:
                print(f"加载分段音频特征缓存失败: {str(e)}")
        
        # 处理每个分段音频
        for i, segment_file in enumerate(segment_files):
            segment_name = os.path.basename(segment_file).split('.')[0]  # 不包含扩展名
            
            # 检查当前分段是否在缓存中
            segment_feature = None
            if segment_name in segment_dict:
                segment_feature = segment_dict[segment_name]['features']
            
            # 如果缓存中没有当前分段的特征，则提取特征
            if segment_feature is None:
                segment_feature = self.feature_extractor.extract_features(segment_file)
                
                # 将新提取的特征添加到字典中
                if segment_features_cache:
                    segment_dict[segment_name] = {
                        'phone_id': phone_id,
                        'segment_id': i,
                        'segment_name': segment_name,
                        'audio_path': segment_file,
                        'features': segment_feature,
                        'extraction_time': torch.tensor(pd.Timestamp.now().timestamp())
                    }
            
            segment_features.append(segment_feature)
        
        # 保存更新后的分段特征字典
        if segment_features_cache:
            torch.save(segment_dict, segment_features_cache)
        
        return original_features, segment_features
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        item = self.data[idx]
        phone_id = item['phone_id']
        label = item.get('label', None)
        speaker = item.get('speaker', None)
        
        # 获取原始音频特征和分段特征
        original_features, segment_features = self._get_audio_segments(item)
        
        # 保存原始标签
        original_label = label
        
        # 标签ID和one-hot处理
        # 明确检查空标签情况
        if label is None or label == '' or label == -1:
            label_id = -1
            label_onehot = torch.zeros(self.num_classes)
        else:
            # 非空标签才进行转换
            str_label = str(label)
            label_id = self.label_to_id.get(str_label, -1)
            label_onehot = self._label_to_onehot(original_label)
        
        # 构建结果字典
        result = {
            'phone_id': phone_id,
            'original_features': original_features,
            'segment_features': segment_features,
            'num_segments': len(segment_features),
            'label': label_id,                  # 数值标签
            'original_label': original_label,   # 原始标签字符串
            'label_onehot': label_onehot,       # one-hot编码标签
            'speaker': speaker,
            'original_audio_file': item['original_audio_file'],
            'segment_files': item['segment_files']
        }
        
        return result

    def collate_fn(self, batch):
        """
        整理批次数据
        
        参数:
            batch: 批次数据列表
            
        返回:
            整理好的批次数据字典
        """
        batch_size = len(batch)
        
        # 获取每个样本的片段数量
        num_segments = [item['num_segments'] for item in batch]
        max_num_segments = max(num_segments)
        
        # 准备批次数据
        phone_ids = []
        original_features = []
        all_segment_features = []
        labels = []
        original_labels = []
        label_onehots = []
        speakers = []
        segment_files = []
        original_audio_files = []
        
        # 收集各项数据
        for item in batch:
            phone_ids.append(item['phone_id'])
            
            # 添加原始音频特征
            original_features.append(item['original_features'])
            
            # 处理片段特征
            item_segment_features = item['segment_features']
            all_segment_features.append(item_segment_features)
            
            # 处理标签 - 确保使用-1表示无标签
            if 'label' in item and item['label'] is not None:
                labels.append(item['label'])
            else:
                labels.append(-1)  # 使用-1表示无标签
                
            # 保存原始标签字符串
            original_labels.append(item.get('original_label', ''))
            
            # 保存one-hot编码标签 - 对于空标签返回全零向量
            if 'label_onehot' in item:
                label_onehots.append(item['label_onehot'])
            else:
                label_onehots.append(torch.zeros(self.num_classes))
            
            # 处理说话者信息
            if item.get('speaker') is not None:
                speakers.append(item['speaker'])
            else:
                speakers.append(-1)  # 使用-1表示无说话者信息
            
            # 收集文件路径
            segment_files.append(item['segment_files'])
            original_audio_files.append(item['original_audio_file'])
        
        # 将标签转换为张量
        labels_tensor = torch.tensor(labels)
        
        # 将one-hot标签堆叠为张量
        label_onehots_tensor = torch.stack(label_onehots)
        
        # 构建返回的批次字典
        batch_dict = {
            'phone_ids': phone_ids,
            'original_features': original_features,
            'segment_features': all_segment_features,
            'num_segments': num_segments,
            'max_num_segments': max_num_segments,
            'labels': labels_tensor,
            'original_labels': original_labels,
            'label_onehots': label_onehots_tensor,  # 添加one-hot标签张量
            'speakers': speakers,
            'segment_files': segment_files,
            'original_audio_files': original_audio_files,
            'batch_size': batch_size,
            'num_classes': self.num_classes  # 添加类别数量
        }
        
        return batch_dict

def create_telemarketing_dataloader(
    data_path,
    feature_extractor=None,
    encoder=None,
    labels_file='labels.csv',
    max_segments=10,
    cache_dir='features_cache',
    batch_size=4,
    shuffle=True,
    num_workers=0,
    system_prompt=None,
    balance_labels=False,  # 新增参数：是否使用标签均衡采样
    front_dense_ratio=0.6,  # 新增参数：前面部分所占比例
    dense_factor=2.0  # 新增参数：前面部分有标签样本的密度倍数
):
    """
    创建电话营销音频数据加载器，用于加载已切分的音频片段
    
    参数:
        data_path: 数据根目录路径
        feature_extractor: 音频特征提取器，如果None则创建默认提取器
        encoder: 音频编码器，用于进一步处理特征
        labels_file: 标签文件名称
        max_segments: 每个对话最多使用的片段数量
        cache_dir: 特征缓存目录
        batch_size: 批次大小
        shuffle: 是否打乱数据顺序
        num_workers: 数据加载的工作线程数量
        system_prompt: 系统提示文本
        balance_labels: 是否使用标签均衡采样，使有标签样本均匀分布
        front_dense_ratio: 前面部分所占总体数据的比例
        dense_factor: 前面部分有标签样本的密度倍数
        
    返回:
        数据加载器
    """
    dataset = AudioSegmentDataset(
        data_path=data_path,
        feature_extractor=feature_extractor,
        encoder=encoder,
        labels_file=labels_file,
        max_segments=max_segments,
        cache_dir=cache_dir,
        system_prompt=system_prompt
    )
    
    # 判断是否使用标签均衡采样
    if balance_labels:
        # 使用LabelBalancedSampler
        sampler = LabelBalancedSampler(
            dataset=dataset,
            batch_size=batch_size,
            front_dense_ratio=front_dense_ratio,
            dense_factor=dense_factor
        )
        
        # 使用自定义采样器时，shuffle参数必须为False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # 使用自定义采样器
            shuffle=False,     # 使用采样器时shuffle必须为False
            collate_fn=dataset.collate_fn,
            num_workers=num_workers
        )
        
        print(f"已启用标签均衡采样，前{int(len(dataset) * front_dense_ratio)}个样本中有标签样本密度为标准的{dense_factor}倍")
    else:
        # 使用常规DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers
        )
    
    return dataloader

class LabelBalancedSampler(Sampler):
    """
    标签均衡采样器，确保有标签的样本均匀分布在批次中，
    并且前面的批次中有更多的有标签样本
    """
    
    def __init__(self, dataset, batch_size=1, front_dense_ratio=0.6, dense_factor=2.0):
        """
        初始化标签均衡采样器
        
        参数:
            dataset: 数据集实例
            batch_size: 批次大小
            front_dense_ratio: 前面部分所占总体数据的比例，这部分会有更密集的有标签样本
            dense_factor: 前面部分有标签样本的密度倍数，相对于平均分布
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.front_dense_ratio = front_dense_ratio
        self.dense_factor = dense_factor
        
        # 获取数据集总样本数
        self.num_samples = len(dataset)
        
        # 将数据集样本分为有标签和无标签两类
        self.labeled_indices = []
        self.unlabeled_indices = []
        
        for i in range(self.num_samples):
            label = dataset.data[i].get('label')
            if label is not None and label != -1 and label != '':
                self.labeled_indices.append(i)
            else:
                self.unlabeled_indices.append(i)
        
        self.num_labeled = len(self.labeled_indices)
        self.num_unlabeled = len(self.unlabeled_indices)
        
        print(f"数据集中有 {self.num_labeled} 个有标签样本, {self.num_unlabeled} 个无标签样本")
        
        # 计算有标签样本的分布
        self.indices = self._distribute_labeled_samples()

    def _distribute_labeled_samples(self):
        """
        计算样本索引顺序，使有标签样本均匀分布，前面部分更密集
        
        返回:
            样本索引列表
        """
        # 计算前面部分和后面部分的样本数
        front_size = int(self.num_samples * self.front_dense_ratio)
        back_size = self.num_samples - front_size
        
        # 计算前面部分应该有多少有标签样本
        # 前面密度更高，所以乘以dense_factor
        front_labeled_ratio = min(1.0, self.dense_factor * self.num_labeled / self.num_samples)
        front_labeled_count = min(self.num_labeled, int(front_size * front_labeled_ratio))
        
        # 后面部分的有标签样本数量
        back_labeled_count = self.num_labeled - front_labeled_count
        
        # 前面部分的无标签样本数量
        front_unlabeled_count = front_size - front_labeled_count
        
        # 后面部分的无标签样本数量
        back_unlabeled_count = self.num_unlabeled - front_unlabeled_count
        
        print(f"前面 {front_size} 个样本中有 {front_labeled_count} 个有标签样本")
        print(f"后面 {back_size} 个样本中有 {back_labeled_count} 个有标签样本")
        
        # 复制标签索引列表，以便我们可以弹出元素
        labeled_indices = self.labeled_indices.copy()
        unlabeled_indices = self.unlabeled_indices.copy()
        
        # 随机打乱索引
        import random
        random.shuffle(labeled_indices)
        random.shuffle(unlabeled_indices)
        
        # 准备结果索引
        result_indices = []
        
        # 前面部分 - 有标签样本和无标签样本交替出现，但有标签样本更多
        front_labeled = labeled_indices[:front_labeled_count]
        front_unlabeled = unlabeled_indices[:front_unlabeled_count]
        
        # 确定前面部分有标签和无标签样本的分布
        # 我们希望大约每2-3个样本中就有1个有标签样本
        # 防止除零错误
        if front_labeled_count > 0:
            step = max(1, int((front_labeled_count + front_unlabeled_count) / front_labeled_count))
        else:
            step = 1
        
        # 构建前面部分的索引
        front_indices = []
        unlabeled_idx = 0
        
        for i, label_idx in enumerate(front_labeled):
            front_indices.append(label_idx)
            
            # 在有标签样本之间插入无标签样本
            unlabeled_to_add = step - 1
            while unlabeled_to_add > 0 and unlabeled_idx < len(front_unlabeled):
                front_indices.append(front_unlabeled[unlabeled_idx])
                unlabeled_idx += 1
                unlabeled_to_add -= 1
        
        # 添加剩余的无标签样本
        while unlabeled_idx < len(front_unlabeled):
            front_indices.append(unlabeled_idx)
            unlabeled_idx += 1
        
        # 后面部分 - 有标签样本更少，更分散
        back_labeled = labeled_indices[front_labeled_count:]
        back_unlabeled = unlabeled_indices[front_unlabeled_count:]
        
        # 确定后面部分有标签和无标签样本的分布
        # 如果后面部分有标签样本很少或没有，就直接添加无标签样本
        if back_labeled_count == 0:
            back_indices = back_unlabeled
        else:
            # 否则均匀分布有标签样本
            back_step = max(1, int((back_labeled_count + back_unlabeled_count) / back_labeled_count))
            
            back_indices = []
            unlabeled_idx = 0
            
            for i, label_idx in enumerate(back_labeled):
                # 先添加无标签样本，再添加有标签样本，这样有标签样本更分散
                unlabeled_to_add = back_step - 1
                while unlabeled_to_add > 0 and unlabeled_idx < len(back_unlabeled):
                    back_indices.append(back_unlabeled[unlabeled_idx])
                    unlabeled_idx += 1
                    unlabeled_to_add -= 1
                
                back_indices.append(label_idx)
            
            # 添加剩余的无标签样本
            while unlabeled_idx < len(back_unlabeled):
                back_indices.append(unlabeled_idx)
                unlabeled_idx += 1
        
        # 组合前面和后面部分的索引
        result_indices = front_indices + back_indices
        
        return result_indices
    
    def __iter__(self):
        """返回索引迭代器"""
        return iter(self.indices)
    
    def __len__(self):
        """返回样本数量"""
        return self.num_samples
