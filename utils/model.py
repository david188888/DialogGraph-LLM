import torch
import librosa
import numpy as np
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# 添加ECAI模块所在的路径
QWEN_ROOT = "/data/shared/Qwen"
if QWEN_ROOT not in sys.path:
    sys.path.insert(0, QWEN_ROOT)

# 导入所需的Transformers组件
from transformers import WhisperFeatureExtractor, AutoConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin, ProcessingKwargs
from transformers.tokenization_utils_base import PaddingStrategy

# 现在导入ECAI模块
from ECAI.qwen2_5_omni_light.model import Qwen2_5OmniTextOnlyModel

# 注意：在新版transformers中，Unpack可能在typing_extensions中
try:
    from transformers.processing_utils import Unpack
except ImportError:
    from typing_extensions import Unpack

# 保留音频编码器输出类
@dataclass
class AudioEncoderOutput:
    """音频编码器输出类，模仿Qwen2_5OmniAudioEncoder的输出结构"""
    last_hidden_state: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class Qwen2_5OmniLightProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
    }

class Qwen2_5OmniAudioProcessor(ProcessorMixin):
    """
    Qwen2.5Omni音频处理器，专用于音频特征提取。
    结合了WhisperFeatureExtractor用于初始特征提取和Qwen2_5OmniAudioEncoder用于音频编码。
    Example:
    path = "file://test.wav"
    audios = librosa.load(path[len("file://") :], sr=16000)[0]
    processor = Qwen2_5OmniAudioProcessor.from_pretrained(model_path)
    inputs = processor(audios=audios, return_tensors="pt")
    """

    attributes = ["feature_extractor", "audio_encoder"]
    feature_extractor_class = "WhisperFeatureExtractor"
    
    def __init__(self, feature_extractor=None, audio_encoder=None):
        self.feature_extractor = feature_extractor
        self.audio_encoder = audio_encoder
        
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        **kwargs
    ):
        """
        从预训练模型路径加载处理器
        
        Args:
            pretrained_model_name_or_path: 预训练模型路径或名称
            **kwargs: 额外参数
            
        Returns:
            Qwen2_5OmniAudioProcessor: 加载好的处理器实例
        """
        # 加载特征提取器
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, 
        )
        
        # 尝试加载音频编码器
        audio_encoder = None

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config = config.thinker_config
        if hasattr(config, "audio_config") and config.audio_config is not None:
            audio_encoder = Qwen2_5OmniAudioEncoder._from_config(
                config.audio_config, attn_implementation=config._attn_implementation
            )
        with torch.no_grad():
                        temp_model = Qwen2_5OmniTextOnlyModel.from_pretrained(
                            pretrained_model_name_or_path,
                            device_map="auto", 
                            torch_dtype=torch.bfloat16, 
                            **{k: v for k, v in kwargs.items() if k != "load_weights"}
                        )
                        
                        # 如果是Qwen2_5OmniThinkerForConditionalGeneration模型
                        if hasattr(temp_model, "audio_tower"):
                            audio_encoder.load_state_dict(temp_model.audio_tower.state_dict())

                        
                        # 清理临时模型
                        del temp_model
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        return cls(feature_extractor=feature_extractor, audio_encoder=audio_encoder)
    
    def __call__(
        self,
        audios: Union[np.ndarray, List[np.ndarray]] = None,
        sampling_rate: Optional[int] = 16000,
        padding: Union[bool, str, PaddingStrategy] = False,
        return_tensors: Optional[str] = "pt",
        **kwargs: Unpack[Qwen2_5OmniLightProcessorKwargs],
    ) -> BatchFeature:
        """
        处理音频输入并提取特征。
        
        Args:
            audios: 输入音频数据，可以是单个音频数组或音频数组列表
            sampling_rate: 音频采样率，默认16kHz
            padding: 是否对输入进行填充
            return_tensors: 返回的张量类型，默认为PyTorch
            **kwargs: 其他参数
            
        Returns:
            包含处理后音频特征的BatchFeature对象
        """
        if audios is None:
            raise ValueError("需要指定`audios`输入进行处理。")
            
        # 确保audios是正确的格式
        if not isinstance(audios, (np.ndarray, list, torch.Tensor)):
            audios_type = type(audios)
            print(f"警告: audios不是预期的类型 (np.ndarray, list, torch.Tensor), 而是 {audios_type}")
            if isinstance(audios, tuple) and len(audios) > 0:  # 如果是元组，可能是librosa.load的输出
                print(f"尝试从元组中提取音频数据")
                audios = audios[0]
                
        # 如果是单个音频而不是列表，将其转换为numpy数组
        if isinstance(audios, list) and len(audios) > 0 and isinstance(audios[0], (float, int)):
            audios = np.array(audios)
            
        # 如果是torch张量，转换为numpy数组
        if isinstance(audios, torch.Tensor):
            audios = audios.cpu().numpy()
            
        # # 打印一下音频数据的信息，便于调试
        # if isinstance(audios, np.ndarray):
        #     # print(f"处理音频数据: 形状={audios.shape}, 类型={audios.dtype}, 值范围=[{np.min(audios):.3f}, {np.max(audios):.3f}]")
        # elif isinstance(audios, list):
        #     print(f"处理音频数据列表，长度={len(audios)}")

        try:
            # 使用WhisperFeatureExtractor提取初始特征
            audio_features = self.feature_extractor(
                audios, 
                sampling_rate=sampling_rate, 
                return_attention_mask=True, 
                padding="max_length",
                return_tensors=return_tensors,
                **kwargs
            )
            
            
            
            # 重命名特征以符合Qwen2.5OmniAudioEncoder的输入格式
            input_features = audio_features.pop("input_features")
            feature_attention_mask = audio_features.pop("attention_mask") if "attention_mask" in audio_features else None
            print(f"shape of input_feature{input_features.shape}")
            outputs = {
                "input_features": input_features,
                "feature_attention_mask": feature_attention_mask,
            }
            
            # 如果提供了audio_encoder，则进一步编码音频特征
            if self.audio_encoder is not None and return_tensors == "pt":
                with torch.no_grad():
                    # 计算音频长度
                    if feature_attention_mask is not None:
                        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)


                    else:
                        audio_feature_lengths = None
                        # 使用_get_feat_extract_output_lengths获取经过音频编码器处理后的长度
                    audio_feat_lengths, audio_output_lengths = self.audio_encoder._get_feat_extract_output_lengths(audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1))
                    feature_lens = (
                        audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                    )

                    # 使用audio_encoder编码音频特征
                    audio_encoder_outputs = self.audio_encoder(
                        input_features,
                        feature_lens=feature_lens,  # 原始特征长度
                        aftercnn_lens=audio_feat_lengths,  # 经过CNN处理后的长度
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    
                    # 添加编码后的特征到输出
                    outputs["audio_encoded_features"] = audio_encoder_outputs.last_hidden_state
                    
                    # 检查并添加attention_mask
                    if hasattr(audio_encoder_outputs, "attention_mask"):
                        outputs["audio_encoded_attention_mask"] = audio_encoder_outputs.attention_mask

                        
                    # 将音频输出长度也添加到输出中，以便后续使用
                    outputs["audio_output_lengths"] = audio_output_lengths
            
            return BatchFeature(
                data=outputs,
                tensor_type=return_tensors,
            )
        except Exception as e:
            print(f"音频处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回空的结果
            return BatchFeature(
                data={
                    "input_features": torch.zeros((1, 80, 3000)) if return_tensors == "pt" else np.zeros((1, 80, 3000)),
                    "feature_attention_mask": torch.zeros((1, 3000)) if return_tensors == "pt" else np.zeros((1, 3000)),
                },
                tensor_type=return_tensors,
            )
        
    @property
    def model_input_names(self):
        names = self.feature_extractor.model_input_names + ["feature_attention_mask"]
        if self.audio_encoder is not None:
            names += ["audio_encoded_features", "audio_encoded_attention_mask"]
        return list(dict.fromkeys(names))  # 去重

# 加载 Qwen2_5OmniAudioProcessor 的辅助函数
def load_qwen_audio_processor(model_path_or_name: str, **kwargs) -> Qwen2_5OmniAudioProcessor:
    """
    从预训练路径加载 Qwen2_5OmniAudioProcessor。

    参数:
        model_path_or_name: 模型路径或名称。
        **kwargs: 传递给 from_pretrained 的其他参数。

    返回:
        Qwen2_5OmniAudioProcessor 实例。
    """
    try:
        logging.info(f"从 {model_path_or_name} 加载 Qwen2_5OmniAudioProcessor...")
        processor = Qwen2_5OmniAudioProcessor.from_pretrained(model_path_or_name, **kwargs)
        return processor
    except Exception as e:
        logging.error(f"加载 Qwen2_5OmniAudioProcessor 失败: {e}")
        raise 