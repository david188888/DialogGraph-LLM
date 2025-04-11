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
from .model import load_qwen_audio_processor
import dataclasses
from typing import Optional

# 默认路径设置
DEFAULT_AUDIO_DIR = "/data/shared/Qwen/data/audio"
DEFAULT_SEGMENTS_DIR = "/data/shared/Qwen/data/segments"
DEFAULT_MODEL_PATH = "/data/shared/Qwen/models/Qwen2.5-Omni-7B"

# DataLoader 配置类
@dataclasses.dataclass
class DataLoaderConfig:
    """DataLoader 配置"""
    data_path: str
    labels_file: str = 'migrated_labels.csv' # 默认使用迁移后的标签
    cache_dir: str = 'features_cache'
    batch_size: int = 8 # 提供一个默认批次大小
    shuffle: bool = True
    num_workers: int = 0
    system_prompt: Optional[str] = None
    balance_labels: bool = False # 默认不使用标签均衡
    front_dense_ratio: float = 0.6
    dense_factor: float = 2.0
    model_path: Optional[str] = DEFAULT_MODEL_PATH # 默认模型路径

class AudioSegmentDataset(Dataset):
    
    def __init__(self, data_path, model_path: str = DEFAULT_MODEL_PATH, 
                 labels_file='migrated_labels.csv', cache_dir='./features_cache', 
                 system_prompt=None, 
                 audio_dir=None, segments_dir=None):
        """
        初始化音频分段数据集
        
        参数:
            data_path: 数据根目录路径
            model_path: 模型路径或名称，用于加载Qwen2_5OmniAudioProcessor
            labels_file: 标签文件名称
            cache_dir: 特征缓存目录
            system_prompt: 系统提示文本
            audio_dir: 音频文件夹路径，不指定则使用 data_path/audio
            segments_dir: 分段文件夹路径，不指定则使用 data_path/segments
        """
        self.data_path = data_path
        self.segments_dir = segments_dir if segments_dir else os.path.join(data_path, 'segments')
        self.audio_dir = audio_dir if audio_dir else os.path.join(data_path, 'audio')
        self.labels_file = os.path.join(data_path, labels_file)
        self.cache_dir = os.path.join(data_path, cache_dir) if cache_dir else None
        self.system_prompt = system_prompt
        
        # 打印使用的路径
        print(f"使用音频目录: {self.audio_dir}")
        print(f"使用分段目录: {self.segments_dir}")
        
        # 加载Qwen2_5OmniAudioProcessor
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
            print(f"未指定模型路径，使用默认路径: {model_path}")
        
        try:
            self.processor = load_qwen_audio_processor(model_path)
            print(f"已从 {model_path} 加载 Qwen2_5OmniAudioProcessor")
        except Exception as e:
            print(f"无法加载音频处理器: {str(e)}")
            raise # 如果处理器无法加载，则无法继续
        
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
        valid_labels = ['A', 'B', 'C', 'D']
        self.num_classes = len(valid_labels)
        for i, label in enumerate(valid_labels):
            self.label_to_id[label] = i
            self.id_to_label[i] = label
        print(f"标签映射: {self.label_to_id}")
        print(f"总共 {self.num_classes} 种类别")

    def _label_to_onehot(self, label):
        """将标签转换为one-hot向量，空标签返回全零向量"""
        if label is None or label == '' or label == -1 or str(label).lower() == 'none':
            return torch.zeros(self.num_classes)
        str_label = str(label)
        if str_label in self.label_to_id:
            label_id = self.label_to_id[str_label]
            one_hot = torch.zeros(self.num_classes)
            one_hot[label_id] = 1.0
            return one_hot
        else:
            print(f"警告：未识别的标签值 '{str_label}'，返回全零向量")
            return torch.zeros(self.num_classes)
            
    def _load_data(self):
        """加载数据，建立音频片段与标签的对应关系，不限制片段数量"""
        self.data = []
        try:
            if os.path.exists(self.labels_file):
                labels_df = pd.read_csv(self.labels_file, header=None)
                labels_dict = {}
                for _, row in labels_df.iterrows():
                    if len(row) >= 3:
                        audio_id = str(row[0])
                        label_value = row[2] if pd.notna(row[2]) and row[2] != 'none' else None
                        labels_dict[audio_id] = {'label': label_value, 'speaker': None}
            else:
                labels_dict = {}
                print(f"警告：找不到标签文件 {self.labels_file}")
            
            if not os.path.exists(self.segments_dir):
                raise FileNotFoundError(f"找不到切分音频目录: {self.segments_dir}")
            
            phone_ids = set()
            for file_name in os.listdir(self.audio_dir):
                if file_name.endswith('.wav'):
                    phone_id = file_name.split('.')[0]
                    phone_ids.add(phone_id)
            
            for phone_id in phone_ids:
                original_audio_file = os.path.join(self.audio_dir, f"{phone_id}.wav")
                if not os.path.exists(original_audio_file):
                    print(f"警告：找不到原始音频文件: {original_audio_file}")
                    continue
                
                segment_files = []
                for file_name in os.listdir(self.segments_dir):
                    if file_name.startswith(f"{phone_id}_") and file_name.endswith('.wav'):
                        segment_files.append(os.path.join(self.segments_dir, file_name))
                
                if not segment_files:
                    print(f"警告：电话ID {phone_id} 没有找到对应的片段文件")
                    continue
                
                try:
                    segment_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
                except Exception as e:
                    print(f"对片段文件排序时出错: {e}，使用文件名排序")
                    segment_files.sort()
                
                label_info = labels_dict.get(phone_id, {})
                label = label_info.get('label', None)
                speaker = label_info.get('speaker', None)
                
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
            print(f"加载了 {len(self.data)} 个电话对话，总共 {total_segments} 个片段")
            print("注意：因为保留了所有片段，建议使用batch_size=1以避免内存问题")
            
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
        
    def _get_audio_features_with_cache(self, audio_path: str, cache_subdir: str, cache_filename: str) -> torch.Tensor:
        """带缓存的音频特征提取逻辑"""
        features = None
        cache_path = None
        
        if self.cache_dir:
            full_cache_subdir = os.path.join(self.cache_dir, cache_subdir)
            os.makedirs(full_cache_subdir, exist_ok=True)
            cache_path = os.path.join(full_cache_subdir, cache_filename)
            
            if os.path.exists(cache_path):
                try:
                    features = torch.load(cache_path)
                    # print(f"特征缓存命中: {cache_path}") # 用于调试
                except Exception as e:
                    print(f"加载特征缓存失败 {cache_path}: {str(e)}")
                    features = None

        if features is None:
            try:
                # 处理音频路径，加载音频数据
                if isinstance(audio_path, str) and audio_path.startswith("file://"):
                    audio_path = audio_path[7:]
                
                # 加载音频
                audio, sr = librosa.load(audio_path, sr=16000)
                
                # 使用Qwen2_5OmniAudioProcessor处理音频
                feature_dict = self.processor(
                    audios=audio,
                    sampling_rate=16000,
                    padding=False,
                    return_tensors="pt"
                )
                
                print(f"特征字典: {feature_dict}")
                
                # 优先使用编码后的特征（如果有）
                if "audio_encoded_features" in feature_dict and feature_dict["audio_encoded_features"].numel() > 0:
                    features = feature_dict["audio_encoded_features"].squeeze(0)
                else:
                    features = feature_dict["input_features"].squeeze(0)
                
            except Exception as e:
                print(f"提取音频特征失败 {audio_path}: {str(e)}")
                features = torch.tensor([]) # 返回空张量
            
            if cache_path and features.numel() > 0: # 只有成功提取才缓存
                try:
                    torch.save(features, cache_path)
                    # print(f"特征缓存已保存: {cache_path}") # 用于调试
                except Exception as e:
                    print(f"保存特征缓存失败 {cache_path}: {str(e)}")
                    
        return features if features is not None else torch.tensor([])

    def _get_audio_segments(self, item):
        """
        获取音频片段的特征和原始音频的特征。
        使用 Qwen2_5OmniAudioProcessor 并应用缓存。
        
        参数:
            item: 数据项，包含原始音频和片段音频的路径
            
        返回:
            原始音频特征和分段特征列表的元组
        """
        phone_id = item['phone_id']
        original_audio_file = item['original_audio_file']
        segment_files = item['segment_files']
        
        # --- 处理原始音频特征 --- 
        original_features = self._get_audio_features_with_cache(
            original_audio_file, 
            "original_features", 
            f"{phone_id}.pt"
        )
        
        # --- 处理分段音频特征 --- 
        segment_features = []
        for i, segment_file in enumerate(segment_files):
            segment_name = os.path.basename(segment_file).split('.')[0]  # 不包含扩展名
            segment_feature = self._get_audio_features_with_cache(
                segment_file, 
                "segment_features", 
                f"{segment_name}.pt"
            )
            
            if segment_feature.numel() > 0: # 只有非空特征才添加
                segment_features.append(segment_feature)
            else:
                print(f"警告：无法提取或加载片段特征，已跳过: {segment_file}")
        
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
        if label is None or label == '' or label == -1:
            label_id = -1
            label_onehot = torch.zeros(self.num_classes)
        else:
            str_label = str(label)
            label_id = self.label_to_id.get(str_label, -1)
            label_onehot = self._label_to_onehot(original_label)
        
        # 构建结果字典
        result = {
            'phone_id': phone_id,
            'original_features': original_features, # [time, feat_dim]
            'segment_features': segment_features, # List of [time, feat_dim]
            'num_segments': len(segment_features),
            'label': label_id,                  
            'original_label': original_label,   
            'label_onehot': label_onehot,       
            'speaker': speaker,
            'original_audio_file': item['original_audio_file'],
            'segment_files': item['segment_files']
        }
        
        return result

    # collate_fn 需要修改以处理不同长度的特征序列
    def collate_fn(self, batch):
        """
        整理批次数据，处理不同长度的特征序列。
        
        参数:
            batch: 批次数据列表
            
        返回:
            整理好的批次数据字典
        """
        batch_size = len(batch)
        
        phone_ids = [item['phone_id'] for item in batch]
        labels = [item.get('label', -1) for item in batch]
        original_labels = [item.get('original_label', '') for item in batch]
        label_onehots = [item.get('label_onehot', torch.zeros(self.num_classes)) for item in batch]
        speakers = [item.get('speaker', -1) for item in batch]
        segment_files = [item['segment_files'] for item in batch]
        original_audio_files = [item['original_audio_file'] for item in batch]
        num_segments = [item['num_segments'] for item in batch]
        max_num_segments = max(num_segments) if num_segments else 0
        
        # --- 处理原始特征填充 --- 
        original_features = [item['original_features'] for item in batch]
        # 获取特征维度（假设所有特征维度相同）
        feat_dim = original_features[0].shape[-1] if batch_size > 0 and original_features[0].numel() > 0 else 80 # 默认 Whisper feat dim
        # 找到最大时间长度
        max_orig_len = max(feat.shape[0] for feat in original_features if feat.numel() > 0) if any(f.numel() > 0 for f in original_features) else 0
        
        padded_original_features = []
        original_attention_mask = []
        
        for feat in original_features:
            if feat.numel() == 0: # 处理空特征
                 padded_feat = torch.zeros((max_orig_len, feat_dim))
                 mask = torch.zeros(max_orig_len, dtype=torch.long)
            else:
                current_len = feat.shape[0]
                padding_len = max_orig_len - current_len
                if padding_len > 0:
                    padded_feat = torch.nn.functional.pad(feat, (0, 0, 0, padding_len), value=0.0) # 填充时间维度
                    mask = torch.cat([torch.ones(current_len, dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)])
                else:
                    padded_feat = feat
                    mask = torch.ones(current_len, dtype=torch.long)
            padded_original_features.append(padded_feat)
            original_attention_mask.append(mask)
            
        original_features_tensor = torch.stack(padded_original_features) if padded_original_features else torch.empty(0)
        original_mask_tensor = torch.stack(original_attention_mask) if original_attention_mask else torch.empty(0)
        
        # --- 处理片段特征填充 --- 
        all_segment_features = [] # List[List[Tensor]]
        all_segment_masks = []    # List[List[Tensor]]
        max_segment_len = 0       # 所有片段中的最大长度

        # 先收集所有片段特征并找到最大长度
        temp_all_segments = []
        for item in batch:
            item_segments = item['segment_features']
            temp_all_segments.extend(item_segments)
        if temp_all_segments:
             max_segment_len = max(seg.shape[0] for seg in temp_all_segments if seg.numel() > 0) if any(s.numel() > 0 for s in temp_all_segments) else 0
        
        # 再次遍历，进行填充和收集
        for item in batch:
            item_segment_features = item['segment_features']
            padded_item_segments = []
            item_segment_masks = []
            
            for i in range(max_num_segments):
                if i < len(item_segment_features) and item_segment_features[i].numel() > 0:
                    feat = item_segment_features[i]
                    current_len = feat.shape[0]
                    padding_len = max_segment_len - current_len
                    if padding_len > 0:
                        padded_feat = torch.nn.functional.pad(feat, (0, 0, 0, padding_len), value=0.0)
                        mask = torch.cat([torch.ones(current_len, dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)])
                    else:
                        padded_feat = feat
                        mask = torch.ones(current_len, dtype=torch.long)
                else: # 如果片段不存在或为空
                    padded_feat = torch.zeros((max_segment_len, feat_dim))
                    mask = torch.zeros(max_segment_len, dtype=torch.long)
                
                padded_item_segments.append(padded_feat)
                item_segment_masks.append(mask)
                
            all_segment_features.append(torch.stack(padded_item_segments) if padded_item_segments else torch.empty(0)) # [max_num_segments, max_segment_len, feat_dim]
            all_segment_masks.append(torch.stack(item_segment_masks) if item_segment_masks else torch.empty(0))       # [max_num_segments, max_segment_len]
            
        # 堆叠批次维度
        segment_features_tensor = torch.stack(all_segment_features) if all_segment_features else torch.empty(0) # [batch, max_num_segments, max_segment_len, feat_dim]
        segment_mask_tensor = torch.stack(all_segment_masks) if all_segment_masks else torch.empty(0)          # [batch, max_num_segments, max_segment_len]
        
        # 将标签转换为张量
        labels_tensor = torch.tensor(labels)
        label_onehots_tensor = torch.stack(label_onehots)
        
        # 构建返回的批次字典
        batch_dict = {
            'phone_ids': phone_ids,
            'original_features': original_features_tensor, # [batch, max_orig_len, feat_dim]
            'original_attention_mask': original_mask_tensor, # [batch, max_orig_len]
            'segment_features': segment_features_tensor,    # [batch, max_num_segments, max_segment_len, feat_dim]
            'segment_attention_mask': segment_mask_tensor,    # [batch, max_num_segments, max_segment_len]
            'num_segments': num_segments,
            'max_num_segments': max_num_segments,
            'labels': labels_tensor,
            'original_labels': original_labels,
            'label_onehots': label_onehots_tensor,  
            'speakers': speakers,
            'segment_files': segment_files,
            'original_audio_files': original_audio_files,
            'batch_size': batch_size,
            'num_classes': self.num_classes 
        }
        
        return batch_dict

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
        
        self.num_samples = len(dataset)
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
        self.indices = self._distribute_labeled_samples()

    def _distribute_labeled_samples(self):
        """计算样本索引顺序，使有标签样本均匀分布，前面部分更密集"""
        front_size = int(self.num_samples * self.front_dense_ratio)
        back_size = self.num_samples - front_size
        front_labeled_ratio = min(1.0, self.dense_factor * self.num_labeled / self.num_samples)
        front_labeled_count = min(self.num_labeled, int(front_size * front_labeled_ratio))
        back_labeled_count = self.num_labeled - front_labeled_count
        front_unlabeled_count = front_size - front_labeled_count
        back_unlabeled_count = self.num_unlabeled - front_unlabeled_count
        print(f"前面 {front_size} 个样本中有 {front_labeled_count} 个有标签样本")
        print(f"后面 {back_size} 个样本中有 {back_labeled_count} 个有标签样本")
        
        labeled_indices = self.labeled_indices.copy()
        unlabeled_indices = self.unlabeled_indices.copy()
        import random
        random.shuffle(labeled_indices)
        random.shuffle(unlabeled_indices)
        
        front_labeled = labeled_indices[:front_labeled_count]
        front_unlabeled = unlabeled_indices[:front_unlabeled_count]
        if front_labeled_count > 0:
            step = max(1, int((front_labeled_count + front_unlabeled_count) / front_labeled_count))
        else:
            step = 1
            
        front_indices = []
        unlabeled_idx = 0
        for i, label_idx in enumerate(front_labeled):
            front_indices.append(label_idx)
            unlabeled_to_add = step - 1
            while unlabeled_to_add > 0 and unlabeled_idx < len(front_unlabeled):
                front_indices.append(front_unlabeled[unlabeled_idx])
                unlabeled_idx += 1
                unlabeled_to_add -= 1
        while unlabeled_idx < len(front_unlabeled):
            front_indices.append(front_unlabeled[unlabeled_idx]) # 修正：添加索引本身而不是计数器
            unlabeled_idx += 1
        
        back_labeled = labeled_indices[front_labeled_count:]
        back_unlabeled = unlabeled_indices[front_unlabeled_count:]
        if back_labeled_count == 0:
            back_indices = back_unlabeled
        else:
            back_step = max(1, int((back_labeled_count + back_unlabeled_count) / back_labeled_count))
            back_indices = []
            unlabeled_idx = 0
            for i, label_idx in enumerate(back_labeled):
                unlabeled_to_add = back_step - 1
                while unlabeled_to_add > 0 and unlabeled_idx < len(back_unlabeled):
                    back_indices.append(back_unlabeled[unlabeled_idx])
                    unlabeled_idx += 1
                    unlabeled_to_add -= 1
                back_indices.append(label_idx)
            while unlabeled_idx < len(back_unlabeled):
                 back_indices.append(back_unlabeled[unlabeled_idx]) # 修正：添加索引本身而不是计数器
                 unlabeled_idx += 1
                 
        result_indices = front_indices + back_indices
        # 确保所有索引都被包含且没有重复
        if len(result_indices) != self.num_samples or len(set(result_indices)) != self.num_samples:
             print(f"警告：标签均衡采样后索引数量 ({len(result_indices)}) 或唯一索引数量 ({len(set(result_indices))}) 与总样本数 ({self.num_samples}) 不匹配。重新生成随机索引。")
             result_indices = list(range(self.num_samples))
             random.shuffle(result_indices)
             
        return result_indices
    
    def __iter__(self):
        """返回索引迭代器"""
        return iter(self.indices)
    
    def __len__(self):
        """返回样本数量"""
        return self.num_samples
