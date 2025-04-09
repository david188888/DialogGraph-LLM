import torch
import torch.nn as nn
import librosa
import logging

class AudioEncoder(nn.Module):
    """简单的音频编码器，用于将梅尔频谱特征转换为更高级的表示"""
    
    def __init__(self, in_features=80, hidden_size=512, out_features=256):
        """
        初始化音频编码器
        
        参数:
            in_features: 输入特征维度（默认为梅尔频谱的频点数）
            hidden_size: 隐藏层大小
            out_features: 输出特征维度
        """
        super(AudioEncoder, self).__init__()
        
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        
        # 定义编码器网络
        self.encoder = nn.Sequential(
            # 卷积层处理时序信息
            nn.Conv1d(in_features, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            
            # 池化层减少序列长度
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size, out_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_features),
            
            # 自适应池化到固定长度
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 全连接层处理最终特征
        self.fc = nn.Linear(out_features, out_features)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为 [batch_size, n_mels, time]
            
        返回:
            编码后的特征，形状为 [batch_size, out_features]
        """
        # 应用编码器网络
        x = self.encoder(x)  # [batch_size, out_features, 1]
        x = x.squeeze(-1)    # [batch_size, out_features]
        x = self.fc(x)       # [batch_size, out_features]
        
        return x
    
    @classmethod
    def create_default_encoder(cls):
        """
        创建默认的音频编码器
        
        返回:
            默认配置的音频编码器实例
        """
        encoder = cls(in_features=80, hidden_size=512, out_features=256)
        return encoder

class AudioFeatureExtractor:
    """音频特征提取器，提取梅尔频谱特征并通过encoder进行处理"""
    
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160, encoder=None):
        """
        初始化音频特征提取器
        
        参数:
            sample_rate: 采样率
            n_mels: 梅尔滤波器组的数量
            n_fft: FFT窗口大小
            hop_length: 帧移
            encoder: 音频编码器，用于将梅尔频谱特征转换为高级特征
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.encoder = encoder
    
    def extract_features(self, audio_path):
        """
        从音频文件中提取特征
        
        参数:
            audio_path: 音频文件路径
            
        返回:
            提取的特征
        """
        try:
            # 如果传入的是带file://前缀的路径，去掉前缀
            if isinstance(audio_path, str) and audio_path.startswith("file://"):
                audio_path = audio_path[7:]
            
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 提取梅尔频谱特征
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # 转换为分贝尺度
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
            
            # 转换为张量
            features = torch.tensor(log_mel_spectrogram).float()
            
            # 如果提供了编码器，则将特征通过编码器处理
            if self.encoder is not None:
                # 假设encoder需要添加批次维度
                features = features.unsqueeze(0)  # [1, n_mels, time]
                with torch.no_grad():
                    encoded_features = self.encoder(features)
                return encoded_features.squeeze(0)  # 移除批次维度
            else:
                return features
                
        except Exception as e:
            logging.error(f"提取特征时出错: {str(e)}")
            # 返回空特征
            return torch.tensor([]) 