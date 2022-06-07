from turtle import forward
import torch 
import math 
import torch.nn as nn 
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)



class IMITERSelfAttention(nn.Module): 
    r"""
    Incorporate two sets of matrices to imitate the fusion between image and text (iit), text and image (iti) 
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.text_seq_len = config.max_position_embeddings 
        num_patch = int(config.image_size / config.patch_size) 
        self.image_seq_len = num_patch * num_patch + 1  # One token for global fusion 

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size 

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob) 

        # Imitation for image and text fusion 
        self.imitate_text_key = nn.Linear(self.image_seq_len, self.text_seq_len) 
        self.imitate_text_value = nn.Linear(self.image_seq_len, self.text_seq_len) 
        self.imitate_image_key = nn.Linear(self.text_seq_len, self.image_seq_len) 
        self.imitate_image_value = nn.Linear(self.text_seq_len, self.image_seq_len)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) 


    def imitate_key_matrix(self, key_layer): 
        t_key_layer = key_layer.permute(0, 1, 3, 2) # (bsz, num_head, head_size, L_text+L_image)
        text_m, image_m = torch.split(t_key_layer, (self.text_seq_len, self.image_seq_len), dim=-1) 
        predict_image_m = self.imitate_image_key(text_m) 
        predict_text_m = self.imitate_text_key(image_m)
        return torch.cat((predict_text_m, predict_image_m), dim=-1).permute(0, 1, 3, 2) 


    def imitate_value_matrix(self, value_layer): 
        t_value_layer = value_layer.permute(0, 1, 3, 2) # (bsz, num_head, head_size, L_text+L_image)
        text_m, image_m = torch.split(t_value_layer, (self.text_seq_len, self.image_seq_len), dim=-1) 
        predict_image_m = self.imitate_image_value(text_m) 
        predict_text_m = self.imitate_text_value(image_m)
        return torch.cat((predict_text_m, predict_image_m), dim=-1).permute(0, 1, 3, 2)


    def compute_imitation_loss(self, key_layer, value_layer, imitate_key_layer, imitate_value_layer): 
        loss = 0 
        loss += F.mse_loss(imitate_key_layer, key_layer) 
        loss += F.mse_loss(imitate_value_layer, value_layer) 
        return loss


    def forward(
            self, 
            hidden_states, 
            attention_mask=None, 
            head_mask=None, 
            output_attentions=False, 
            predicted_usage=True, # Use the imitated key, value forward 
        ):

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


        outputs = (context_layer, imitation_loss, attention_probs) if output_attentions else (context_layer, imitation_loss,)

        return outputs  



class IMITERSelfOutput(nn.Module):
    """
    The residual connection is defined in IMITERLayer instead of here (as is the case with other models), due to the
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



class IMITERAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = IMITERSelfAttention(config)
        self.output = IMITERSelfOutput(config)
        

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add loss if we output them
        return outputs 



