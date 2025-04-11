import logging
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import AutoConfig, WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
from .modeling_qwen2_5omni_light import Qwen2_5OmniTextOnlyModel
# 注意：在新版transformers中，Unpack可能在typing_extensions中
try:
    from transformers.processing_utils import Unpack
except ImportError:
    from typing_extensions import Unpack

class Qwen2_5OmniLightProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
    }

class Qwen2_5OmniLightProcessor(ProcessorMixin):
    """
    简化版的Qwen2.5Omni处理器，仅处理文本和音频输入。
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    valid_kwargs = ["chat_template"]

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        self.audio_token = "<|AUDIO|>"
        self.audio_bos_token = "<|audio_bos|>"
        self.audio_eos_token = "<|audio_eos|>"
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audios: Union[np.ndarray, List[np.ndarray]] = None,
        sampling_rate: Optional[int] = 16000,
        padding: Union[bool, str, PaddingStrategy] = False,
        **kwargs: Unpack[Qwen2_5OmniLightProcessorKwargs],
    ) -> BatchFeature:
        """
        处理文本和音频输入。
        """
        output_kwargs = self._merge_kwargs(
            Qwen2_5OmniLightProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            padding=padding,
            **kwargs,
        )

        if audios is not None:
            audios_inputs = self.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, padding="max_length", **kwargs
            )
            audios_inputs["feature_attention_mask"] = audios_inputs.pop("attention_mask")
            audios_inputs["input_features"] = audios_inputs.pop("input_features")
            input_lengths = (audios_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
            audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            audios_inputs = {}
            audio_lengths = None

        if text is None:
            raise ValueError("You need to specify a `text` input to process.")

        if not isinstance(text, list):
            text = [text]

        audio_index = 0
        for i in range(len(text)):
            positions = []
            start = 0
            while True:
                pos = text[i].find(self.audio_token, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + len(self.audio_token)
            
            positions.sort()
            for pos in positions:
                if audios is not None:
                    text[i] = text[i].replace(
                        self.audio_token,
                        "<|audio_placeholder|>" * audio_lengths[audio_index],
                        1,
                    )
                    audio_index += 1
            
            text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **audios_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + ["feature_attention_mask"]
            )
        )
        
        



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
        
        # 加载音频编码器配置
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config = config.thinker_config
        if hasattr(config, "audio_config") and config.audio_config is not None:
            audio_encoder = Qwen2_5OmniAudioEncoder._from_config(
                config.audio_config, attn_implementation=config._attn_implementation
            )
        with torch.no_grad():
                model = Qwen2_5OmniTextOnlyModel.from_pretrained(
                    pretrained_model_name_or_path,
                    device_map="auto", 
                    torch_dtype=torch.bfloat16,
                    **{k: v for k, v in kwargs.items() if k != "load_weights"}
                )
                        
                # 如果是Qwen2_5OmniThinkerForConditionalGeneration模型
                if hasattr(model, "audio_tower"):
                    audio_encoder.load_state_dict(model.audio_tower.state_dict())

                        
                # 清理临时模型
                del model
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return cls(feature_extractor=feature_extractor, audio_encoder=audio_encoder)
        
        
    
    
    def __call__(
        self,
        audios: Union[np.ndarray, List[np.ndarray]] = None,
        sampling_rate: Optional[int] = 16000,
        padding: Union[bool, str, PaddingStrategy] = False ,
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

        output_kwargs = self._merge_kwargs(
            Qwen2_5OmniLightProcessorKwargs,
            feature_extractor_init_kwargs=self.feature_extractor.init_kwargs if hasattr(self.feature_extractor, "init_kwargs") else {},
            padding=padding,
            **kwargs,
        )
        
        # 使用WhisperFeatureExtractor提取初始特征
        audio_features = self.feature_extractor(
            audios, 
            sampling_rate=sampling_rate, 
            return_attention_mask=True, 
            padding="max_length" ,
            return_tensors=return_tensors,
            **kwargs
        )
        
        # 重命名特征以符合Qwen2.5OmniAudioEncoder的输入格式
        input_features = audio_features.pop("input_features")
        feature_attention_mask = audio_features.pop("attention_mask") if "attention_mask" in audio_features else None
        
        outputs = {
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
        }
        
        # 如果提供了audio_encoder，则进一步编码音频特征
        if self.audio_encoder is not None and return_tensors == "pt":
            with torch.no_grad():
                # 计算音频长度
                if feature_attention_mask is not None:
                    feature_lens = feature_attention_mask.sum(-1)
                    audio_feat_lengths, audio_output_lengths = self.audio_encoder._get_feat_extract_output_lengths(feature_lens)
                else:
                    feature_lens = None
                    audio_feat_lengths = None
                
                # 使用audio_encoder编码音频特征
                audio_encoder_outputs = self.audio_encoder(
                    input_features,
                    feature_lens=feature_lens,
                    aftercnn_lens=audio_feat_lengths,
                    output_hidden_states=False,
                    return_dict=True,
                )
                
                # 添加编码后的特征到输出
                outputs["audio_encoded_features"] = audio_encoder_outputs.last_hidden_state
                if hasattr(audio_encoder_outputs, "attention_mask"):
                    outputs["audio_encoded_attention_mask"] = audio_encoder_outputs.attention_mask
        
        return BatchFeature(
            data=outputs,
            tensor_type=return_tensors,
        )
        
    @property
    def model_input_names(self):
        names = self.feature_extractor.model_input_names + ["feature_attention_mask"]
        if self.audio_encoder is not None:
            names += ["audio_encoded_features", "audio_encoded_attention_mask"]
        return list(dict.fromkeys(names))  # 去重