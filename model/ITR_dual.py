# Referring for CLIP with similar structure 
import torch 
import torch.nn as nn 

from transformers import PreTrainedModel
from .IMITER import IMITEREmbeddings, IMITERPooler, IMITEREncoder
from .model_config import ClipConfig
from .loss import contrastive_loss, accuracy_compute 


class ClipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ClipConfig
    base_model_prefix = "vilt"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None: 
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, IMITEREncoder):
            module.gradient_checkpointing = value



class ClipModel(ClipPreTrainedModel): 

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = IMITEREmbeddings(config)
        self.visual_encoder = IMITEREncoder(config) 
        self.text_encoder = IMITEREncoder(config)

        self.visual_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_pooler = IMITERPooler(config) if add_pooling_layer else None
        self.text_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.text_pooler = IMITERPooler(config) if add_pooling_layer else None

        self.text_seq_len = config.max_position_embeddings 
        num_patch = int(config.image_size / config.patch_size) 
        self.image_seq_len = num_patch * num_patch 

        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)
        # Initialize weights and apply final processing
        # self.post_init()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        pixel_mask=None,
        head_mask=None,
        inputs_embeds=None,
        image_embeds=None,
        image_token_type_idx=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 
        

        if input_ids is not None:
            image_modality = False
            input_shape = input_ids.size() 
        

            batch_size, seq_length = input_shape

            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=input_ids.device)

            embedding_output, attention_mask = self.embeddings.step(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                image_token_type_idx=image_token_type_idx, 
            )


            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)

            encoder_outputs = self.text_encoder.step(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                image_modality=image_modality,
                predicted_usage=False
            )
            sequence_output = encoder_outputs[0] 
   
            sequence_output = self.text_layernorm(sequence_output)
            text_embeds = self.text_pooler(sequence_output)[:, 0, :]


        if pixel_values is not None: 
            image_modality = True
            batch_size, num_channels, height, width = pixel_values.shape
            if pixel_mask is None:
                pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)
            input_shape = (batch_size, 144)

            embedding_output, attention_mask = self.embeddings.step(
                pixel_values = pixel_values,
                pixel_mask = pixel_mask,
                image_token_type_idx=image_token_type_idx, 
            )

            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)

            encoder_outputs = self.visual_encoder.step(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                image_modality=image_modality,
                predicted_usage=False
            )
            sequence_output = encoder_outputs[0] 
   
            sequence_output = self.visual_layernorm(sequence_output)
            image_embeds = self.visual_pooler(sequence_output)[:, 0, :]

        if return_loss: 
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
            logits_per_image = logits_per_text.t()

            loss = None
        
            # logits_per_text.size() == logits_per_image.size() == (bsz, bsz) 
            loss = contrastive_loss(logits_per_text)
            accuracy = accuracy_compute(logits_per_text) 

        output = (logits_per_image, logits_per_text, text_embeds, image_embeds, )
        return ((loss, accuracy, ) + output) if loss is not None else output
