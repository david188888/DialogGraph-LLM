import torch
from torch import nn
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass

from transformers import PreTrainedModel
# 正确导入GenerationMixin
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import ModelOutput

# 导入Qwen2.5Omni相关配置和模型
from transformers import (
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniConfig
)
from transformers import (
    Qwen2_5OmniPreTrainedModel,
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniThinkerModel
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder

@dataclass
class Qwen2_5OmniLightCausalLMOutputWithPast(ModelOutput):
    """
    为简化版的Qwen2.5OmniLight模型设计的输出类，只支持音频和文本模态。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            语言建模损失(用于下一个token预测)。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            语言建模头的预测分数(SoftMax之前每个词汇token的分数)。
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            包含预计算的隐藏状态(self-attention块中的key和value)的元组，可用于加速序列解码。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            每层输出的隐藏状态元组，形状为`(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            每层注意力权重的元组，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
        attention_mask (`torch.FloatTensor`, *optional*):
            注意力掩码，用于更新attention_mask和position_ids。
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            序列长度和多模态rope之间的rope索引差异。
            
            
        数据容器类,用于封装model output
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    rope_deltas: Optional[torch.LongTensor] = None

class Qwen2_5OmniLightForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    """
    简化版的Qwen2.5Omni模型，只支持音频和文本模态。
    
    这个模型移除了图像和视频处理相关的代码，保留了原始Qwen2_5OmniThinkerForConditionalGeneration
    处理音频和文本的功能。
    """
    
    config_class = Qwen2_5OmniThinkerConfig
    _no_split_modules = ["Qwen2_5OmniAudioEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        
        # 正确初始化子模型，使用_from_config方法
        self.audio_tower = None
        if hasattr(config, "audio_config") and config.audio_config is not None:
            self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(
                config.audio_config, attn_implementation=config._attn_implementation
            )
        
        self.vocab_size = config.text_config.vocab_size
        
        # 初始化主模型

        self.model = Qwen2_5OmniThinkerModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: Optional[bool] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2_5OmniLightCausalLMOutputWithPast]:
        """
        简化版的forward方法，只处理音频和文本输入。
        
        Args:
            input_ids: 输入ID
            input_features: 音频特征输入
            attention_mask: 注意力掩码
            feature_attention_mask: 特征注意力掩码
            audio_feature_lengths: 音频特征长度
            position_ids: 位置ID
            past_key_values: 过去的键值对
            inputs_embeds: 输入嵌入
            rope_deltas: Rope索引差异
            labels: 标签(用于计算损失)
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否以字典形式返回
            cache_position: 缓存位置
            
        Returns:
            模型输出
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None
            
        # 处理位置编码
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    None,
                    None,
                    attention_mask,
                    False,  # use_audio_in_video设为False
                    audio_feature_lengths,
                    None,
                )
                rope_deltas = rope_deltas - delta0
            else:
                # 计算后续生成token的位置编码
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            embeds_to_talker = inputs_embeds.clone()

            # 2. Merge text and audios
            if input_ids.shape[1] != 1:
                if input_features is not None:
                    audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                        audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                    )
                    feature_lens = (
                        audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                    )
                    audio_outputs = self.audio_tower(
                        input_features,
                        feature_lens=feature_lens,
                        aftercnn_lens=audio_feat_lengths,
                    )
                    audio_features = audio_outputs.last_hidden_state
                    if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                        raise ValueError("length of audio_features should match audio_output_lengths")
                    audio_mask = (input_ids == self.config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
                    embeds_to_talker = embeds_to_talker.masked_scatter(audio_mask, torch.zeros_like(audio_features))

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + ((embeds_to_talker, outputs[0])) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniLightCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(embeds_to_talker, outputs.hidden_states),
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        input_features=None,
        feature_attention_mask=None,
        **kwargs,
    ):
        """
        为生成准备输入 - 简化版，只处理音频和文本。
        """
        # 使用父类方法
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs
        )
        

        model_inputs["position_ids"] = None
        
        return model_inputs

    # 使用父类的_update_model_kwargs_for_generation方法
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update attention_mask
        if getattr(outputs, "attention_mask", None) is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs
        

class Qwen2_5OmniTextOnlyModel(Qwen2_5OmniPreTrainedModel):
    """
    优化版的Qwen2.5Omni纯文本输出模型。
    专注于高效处理多模态输入并只生成文本输出。
    """
    config_class = Qwen2_5OmniConfig

    def __init__(self, config):
        super().__init__(config)
        # 只初始化Thinker组件
        self.thinker = Qwen2_5OmniLightForConditionalGeneration(config.thinker_config)
        
    @classmethod
    def can_generate(cls) -> bool:
        return True
        
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        use_audio_in_video=False,
        thinker_max_new_tokens=1024,
        **kwargs,
    ):
        """
        从多模态输入生成文本输出。
        
        Args:
            input_ids: 输入ID，应该从processor获取
            use_audio_in_video: 是否使用视频中的音频轨道
            thinker_max_new_tokens: 生成的最大新token数量
            **kwargs: 其他参数，将传递给thinker的generate方法
            
        Returns:
            生成的文本token序列
        """
        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        
        # 处理特殊输入和参数
        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_"):]] = value
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            else:
                shared_kwargs[key] = value
                
        # 合并参数
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
                
        # 使用Thinker生成文本
        return self.thinker.generate(
            input_ids=input_ids,
            **thinker_kwargs,
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """从预训练模型加载TextOnly版本，只加载Thinker部分"""
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)  

