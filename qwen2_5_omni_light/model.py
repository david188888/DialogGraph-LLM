import torch
from torch import nn
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass

from transformers import PreTrainedModel
# Correctly import GenerationMixin
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import ModelOutput

# Import Qwen2.5Omni related configs and models
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
    Output class for the simplified Qwen2.5OmniLight model, supporting only audio and text modalities.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple containing precomputed hidden states (key and value in self-attention blocks), used to speed up sequence decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of hidden states from each layer, shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of attention weights from each layer, shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        attention_mask (`torch.FloatTensor`, *optional*):
            Attention mask, used to update attention_mask and position_ids.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            Rope index difference between sequence length and multimodal rope.

        Data container class for model output.
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
    Simplified version of the Qwen2.5Omni model, supporting only audio and text modalities.

    This model removes image and video processing code, retaining the original Qwen2_5OmniThinkerForConditionalGeneration's audio and text processing functionality.
    """
    
    config_class = Qwen2_5OmniThinkerConfig
    _no_split_modules = ["Qwen2_5OmniAudioEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        
        # Correctly initialize submodules using _from_config
        self.audio_tower = None
        if hasattr(config, "audio_config") and config.audio_config is not None:
            self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(
                config.audio_config, attn_implementation=config._attn_implementation
            )
        
        self.vocab_size = config.text_config.vocab_size
        
        # Initialize main model
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
        graph_audio_features: Optional[torch.FloatTensor] = None,
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
        Simplified forward method, only processes audio and text inputs.
        
        Args:
            input_ids: Input IDs
            input_features: Audio feature input
            attention_mask: Attention mask
            feature_attention_mask: Feature attention mask
            audio_feature_lengths: Audio feature lengths
            position_ids: Position IDs
            past_key_values: Past key-value pairs
            inputs_embeds: Input embeddings
            rope_deltas: Rope index difference
            labels: Labels (for loss computation)
            use_cache: Whether to use cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return as a dictionary
            cache_position: Cache position
        Returns:
            Model output
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
                if graph_audio_features is not None:
                    image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    graph_audio_features = graph_audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, graph_audio_features)


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
        Prepare inputs for generation - simplified version, only processes audio and text.
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
    Optimized Qwen2.5Omni text-only output model.
    Focuses on efficiently handling multimodal input and generating only text output.
    """
    config_class = Qwen2_5OmniConfig

    def __init__(self, config):
        super().__init__(config)
        # Only initialize the Thinker component
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
        Generate text output from multimodal input.
        
        Args:
            input_ids: Input IDs, should be obtained from processor
            use_audio_in_video: Whether to use audio track in video
            thinker_max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Other arguments, passed to the thinker's generate method
        Returns:
            Generated text token sequence
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
        """Load TextOnly version from pretrained model, only loads Thinker part"""
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)  

