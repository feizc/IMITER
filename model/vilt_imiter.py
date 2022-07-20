import collections.abc
import math 
from typing import List, Optional, Tuple
import torch.nn.functional as F

import torch
import torch.utils.checkpoint
from torch import Tensor 
from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
)

from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer

from .model_config import ViltConfig 



class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.4") or use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)




class ViltForImagesAndTextClassificationOutput(ModelOutput):
    """
    Class for outputs of [`ViltForImagesAndTextClassification`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`List[tuple(torch.FloatTensor)]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            List of tuples of `torch.FloatTensor` (one for each image-text pair, each tuple containing the output of
            the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`List[tuple(torch.FloatTensor)]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            List of tuples of `torch.FloatTensor` (one for each image-text pair, each tuple containing the attention
            weights of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the
            attention softmax, used to compute the weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[List[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[List[Tuple[torch.FloatTensor]]] = None


# Copied from transformers.models.vit.modeling_vit.to_2tuple
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class ViltEmbeddings(nn.Module):
    """
    Construct the text and patch embeddings.

    Text embeddings are equivalent to BERT embeddings.

    Patch embeddings are equivalent to ViT embeddings.
    """

    def __init__(self, config):
        super().__init__()

        # text embeddings
        self.text_embeddings = TextEmbeddings(config) 

        # patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # modality type (text/patch) embeddings
        self.token_type_embeddings = nn.Embedding(config.modality_type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def visual_embed(self, pixel_values, pixel_mask, max_image_length=200):
        _, _, ph, pw = self.patch_embeddings.projection.weight.shape

        x = self.patch_embeddings(pixel_values)
        x_mask = pixel_mask[:, None, :, :].float()
        x_mask = nn.functional.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]

        batch_size, num_channels, height, width = x.shape
        patch_dim = self.config.image_size // self.config.patch_size
        spatial_pos = self.position_embeddings[:, 1:, :].transpose(1, 2).view(1, num_channels, patch_dim, patch_dim)
        pos_embed = torch.cat(
            [
                nn.functional.pad(
                    nn.functional.interpolate(
                        spatial_pos,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    (0, width - w, 0, height - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )

        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        patch_index = torch.stack(
            torch.meshgrid(torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1]), indexing="ij"), dim=-1
        )
        patch_index = patch_index[None, None, :, :, :]
        patch_index = patch_index.expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
        patch_index = patch_index.flatten(1, 3)
        x_mask = x_mask.flatten(1)

        if max_image_length < 0 or max_image_length is None or not isinstance(max_image_length, int):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            effective_resolution = x_h * x_w
            max_image_length = effective_resolution.max()
        else:
            effective_resolution = x_h * x_w
            max_image_length = min(effective_resolution.max(), max_image_length)

        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_length - v for v in valid_nums]

        select = list()
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_length)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(torch.ones(nv).float(), p, replacement=True)
                select.append(torch.cat([valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0))

        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(batch_size, -1)
        patch_index = patch_index[select[:, 0], select[:, 1]].view(batch_size, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat(
            (self.position_embeddings[:, 0, :][:, None, :].expand(batch_size, -1, -1), pos_embed), dim=1
        )
        x = x + pos_embed
        x = self.dropout(x)

        x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)

        return x, x_mask, (patch_index, (height, width))

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask,
        inputs_embeds,
        image_embeds,
        image_token_type_idx=1,
    ):
        # PART 1: text embeddings
        text_embeds = self.text_embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1).to(attention_mask), attention_mask], dim=1)


        # PART 2: patch embeddings (with interpolated position encodings)
        if image_embeds is None:
            image_embeds, image_masks, patch_index = self.visual_embed(
                pixel_values, pixel_mask, max_image_length=self.config.max_image_length
            )
        else:
            image_masks = pixel_mask.flatten(1)

        # PART 3: add modality type embeddings
        # 0 indicates text, 1 indicates image, 2 is optionally used when a second image is provided (NLVR2)
        if image_token_type_idx is None:
            image_token_type_idx = 1
        text_embeds = text_embeds + self.token_type_embeddings(
            torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device)
        )
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx, dtype=torch.long, device=text_embeds.device)
        )

        # PART 4: concatenate
        embeddings = torch.cat([text_embeds, image_embeds], dim=1)
        masks = torch.cat([attention_mask, image_masks], dim=1)

        return embeddings, masks


class TextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 1, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings + 1).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1] 
        batch_size = input_shape[0]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length + 1] 

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        inputs_embeds = torch.cat((cls_tokens, inputs_embeds), dim=1)
        input_shape = inputs_embeds.size()[:-1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length + 1]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length + 1)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        x = self.projection(pixel_values)
        return x


class ViltSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        
        self.text_seq_len = config.max_position_embeddings 
        num_patch = int(config.image_size / config.patch_size) 
        self.image_seq_len = num_patch * num_patch 

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob) 

        # Imitation for image and text fusion 
        self.imitate_text_key = nn.Linear(self.image_seq_len + 1, self.text_seq_len + 1) 
        self.imitate_text_value = nn.Linear(self.image_seq_len + 1, self.text_seq_len + 1) 
        self.imitate_image_key = nn.Linear(self.text_seq_len + 1, self.image_seq_len + 1) 
        self.imitate_image_value = nn.Linear(self.text_seq_len + 1, self.image_seq_len + 1)



    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) 

    
    def imitate_key_matrix(self, key_layer, image_modality=True, text_modality=True): 
        t_key_layer = key_layer.permute(0, 1, 3, 2) # (bsz, num_head, head_size, {L_text + 1 +L_image+1, L_text+1, L_image+1}) 

        if image_modality == True and text_modality == True: 
            text_m, image_m = torch.split(t_key_layer, (self.text_seq_len + 1, self.image_seq_len + 1), dim=-1) 
            predict_image_m = self.imitate_image_key(text_m)
            predict_text_m = self.imitate_text_key(image_m)
            imitate_key = torch.cat((predict_text_m, predict_image_m), dim=-1).permute(0, 1, 3, 2) 
        elif image_modality == True and text_modality == False: 
            predict_text_m = self.imitate_text_key(t_key_layer) 
            imitate_key = torch.cat((predict_text_m, t_key_layer), dim=-1).permute(0, 1, 3, 2) 
        elif image_modality == False and text_modality == True: 
            predict_image_m = self.imitate_image_key(t_key_layer) 
            imitate_key = torch.cat((t_key_layer, predict_image_m), dim=-1).permute(0, 1, 3, 2) 
        else: 
            raise ValueError("At least one modality information is required for imitation!") 
        return imitate_key 


    def imitate_value_matrix(self, value_layer, image_modality=True, text_modality=True): 
        t_value_layer = value_layer.permute(0, 1, 3, 2) # (bsz, num_head, head_size, {L_text+1 + L_image+1, L_text+1, L_image+1})
        
        if image_modality == True and text_modality == True: 
            text_m, image_m = torch.split(t_value_layer, (self.text_seq_len + 1, self.image_seq_len + 1), dim=-1) 
            predict_image_m = self.imitate_image_value(text_m)
            predict_text_m = self.imitate_text_value(image_m)
            imitate_value = torch.cat((predict_text_m, predict_image_m), dim=-1).permute(0, 1, 3, 2)
        elif image_modality == True and text_modality == False: 
            predict_text_m = self.imitate_text_value(t_value_layer) 
            imitate_value = torch.cat((predict_text_m, t_value_layer), dim=-1).permute(0, 1, 3, 2) 
        elif image_modality == False and text_modality == True: 
            predict_image_m = self.imitate_image_key(t_value_layer) 
            imitate_value = torch.cat((t_value_layer, predict_image_m), dim=-1).permute(0, 1, 3, 2)  
        else: 
            raise ValueError("At least one modality information is required for imitation!") 
        return imitate_value 


    def compute_imitation_loss(self, key_layer, value_layer, imitate_key_layer, imitate_value_layer): 
        loss = 0 
        loss += F.mse_loss(imitate_key_layer, key_layer) 
        loss += F.mse_loss(imitate_value_layer, value_layer) 
        return loss



    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, predicted_usage=True ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        imitate_key_layer = self.imitate_key_matrix(key_layer) 
        imitate_value_layer = self.imitate_value_matrix(value_layer) 

        imitation_loss = self.compute_imitation_loss(key_layer, value_layer, imitate_key_layer, imitate_value_layer) 

        if predicted_usage == True: 
            key_layer = imitate_key_layer 
            value_layer = imitate_value_layer

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, imitation_loss, attention_probs) if output_attentions else (context_layer, imitation_loss)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Vilt
class ViltSelfOutput(nn.Module):
    """
    The residual connection is defined in ViltLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViltAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViltSelfAttention(config)
        self.output = ViltSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->Vilt
class ViltIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = GELUActivation()
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->Vilt
class ViltOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViltLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViltAttention(config)
        self.intermediate = ViltIntermediate(config)
        self.output = ViltOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViLT, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViLT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViltEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViltLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None 
        imitation_loss = 0 

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0] 
            imitation_loss += layer_outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        
        
        if not return_dict:
            return tuple(v for v in [hidden_states, imitation_loss, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=imitation_loss,
            attentions=all_self_attentions,
        )


class ViltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViltConfig
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
        if isinstance(module, ViltEncoder):
            module.gradient_checkpointing = value


VILT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViltConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VILT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViltFeatureExtractor`]. See
            [`ViltFeatureExtractor.__call__`] for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
            `What are attention masks? <../glossary.html#attention-mask>`__

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
            Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `pixel_values` into patch embeddings.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

VILT_IMAGES_AND_TEXT_CLASSIFICATION_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_images, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViltFeatureExtractor`]. See
            [`ViltFeatureExtractor.__call__`] for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, num_images, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
            `What are attention masks? <../glossary.html#attention-mask>`__

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
            Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `pixel_values` into patch embeddings.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""



class ViltModel(ViltPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = ViltEmbeddings(config)
        self.encoder = ViltEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViltPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.text_embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


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
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltModel
        >>> from PIL import Image
        >>> import requests

        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        >>> model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

        >>> inputs = processor(image, text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch_size, num_channels, height, width = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, attention_mask = self.embeddings(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask,
            inputs_embeds,
            image_embeds,
            image_token_type_idx=image_token_type_idx,
        )


        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # return dict
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_txt, pooled_img = self.pooler(sequence_output)
        #pooled_output = torch.cat((pooled_txt, pooled_img),dim=1)
        imitation_loss = encoder_outputs[1]

        if not return_dict:
            return (sequence_output, pooled_txt, pooled_img, imitation_loss) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=(pooled_txt, pooled_img, imitation_loss),
            #pooled_txt = pooled_txt,
            #pooled_img = pooled_img,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViltPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh() 

        self.dense_img = nn.Linear(config.hidden_size, config.hidden_size) 
        self.activation_img = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # text
        first_token_tensor = hidden_states[:, 0] 
        # image 
        second_token_tensor = hidden_states[:, 41] 
        # combine_token_tensor = torch.cat((first_token_tensor, second_token_tensor), dim=1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        
        pooled_output_img = self.dense_img(second_token_tensor) 
        pooled_output_img = self.activation_img(pooled_output_img)
        return pooled_output, pooled_output_img



class ViltForMaskedLM(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vilt = ViltModel(config)
        self.mlm_score = ViltMLMHead(config)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_output_embeddings(self):
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_score.decoder = new_embeddings


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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in *[-100, 0, ...,
            config.vocab_size]* (see *input_ids* docstring) Tokens with indices set to *-100* are ignored (masked), the
            loss is only computed for the tokens with labels in *[0, ..., config.vocab_size]*

        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForMaskedLM
        >>> import requests
        >>> from PIL import Image
        >>> import re

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a bunch of [MASK] laying on a [MASK]."

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        >>> model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**encoding)

        >>> tl = len(re.findall("\[MASK\]", text))
        >>> inferred_token = [text]

        >>> # gradually fill in the MASK tokens, one by one
        >>> with torch.no_grad():
        ...     for i in range(tl):
        ...         encoded = processor.tokenizer(inferred_token)
        ...         input_ids = torch.tensor(encoded.input_ids).to(device)
        ...         encoded = encoded["input_ids"][0][1:-1]
        ...         outputs = model(input_ids=input_ids, pixel_values=pixel_values)
        ...         mlm_logits = outputs.logits[0]  # shape (seq_len, vocab_size)
        ...         # only take into account text features (minus CLS and SEP token)
        ...         mlm_logits = mlm_logits[1 : input_ids.shape[1] - 1, :]
        ...         mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
        ...         # only take into account text
        ...         mlm_values[torch.tensor(encoded) != 103] = 0
        ...         select = mlm_values.argmax().item()
        ...         encoded[select] = mlm_ids[select].item()
        ...         inferred_token = [processor.decode(encoded)]

        >>> selected_token = ""
        >>> encoded = processor.tokenizer(inferred_token)
        >>> processor.decode(encoded.input_ids[0], skip_special_tokens=True)
        a bunch of cats laying on a couch.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        # split up final hidden states into text and image features
        text_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        text_features, _ = (sequence_output[:, :text_seq_len], sequence_output[:, text_seq_len:])

        mlm_logits = self.mlm_score(text_features)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (mlm_logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ViltPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = GELUActivation()
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ViltMLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        self.transform = ViltPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x


class ViltForQuestionAnswering(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vilt = ViltModel(config)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.num_labels),
        )

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the visual question answering loss. This tensor must be either a one-hot encoding of
            all answers that are applicable for a given example in the batch, or a soft encoding indicating which
            answers are applicable, where 1.0 is the highest score.

        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForQuestionAnswering
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are there?"

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        >>> model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**encoding)
        >>> logits = outputs.logits
        >>> idx = logits.argmax(-1).item()
        >>> print("Predicted answer:", model.config.id2label[idx])
        Predicted answer: 2
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels) * labels.shape[1]
            # see https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class ViltForImageAndTextRetrieval(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vilt = ViltModel(config)
        # self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592,) 
        # Classifier head
        self.rank_output = nn.Linear(config.hidden_size * 2, 1)
        print('vilt cos loss new version')
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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels are currently not supported.

        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        >>> model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> scores = dict()
        >>> for text in texts:
        ...     encoding = processor(image, text, return_tensors="pt")
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, :].item()
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #pooler_output = outputs.pooler_output if return_dict else outputs[1] 
        text_f = outputs.pooler_output[0] if return_dict else outputs[1]
        image_f = outputs.pooler_output[1] if return_dict else outputs[2] 
        imitation_loss = outputs.pooler_output[2] if return_dict else outputs[3] 
        
        pooler_output = torch.cat((text_f, image_f), dim=-1)
        logits = self.rank_output(pooler_output)
        
        #image_f = image_f / image_f.norm(dim=-1, keepdim=True)
        #text_f = text_f / text_f.norm(dim=-1, keepdim=True) 
        #score = self.cos_loss(image_f, text_f) # (bsz, ) 
        #logit_scale = self.logit_scale.exp() 
        #score = image_f * text_f * logit_scale
        #logits = torch.sum(score, dim=-1)
        # print(logits.size())
        loss = imitation_loss
        if labels is not None:
            raise NotImplementedError("Training is not yet supported.")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )