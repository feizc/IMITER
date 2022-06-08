import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from einops import rearrange 


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist 


def compute_image_text_retrieval_loss(model, batch, train_config, imitation_loss=False): 
    _bs, _c, _h, _w = batch['image'][0].shape 
    false_len = train_config.draw_false_text 
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )
    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w) 

    infer = model(
        pixel_values = rearrange(images, "bs fs c h w -> (bs fs) c h w"),
        input_ids = rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
        attention_mask = rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
    )
    score = infer[0][:, 0] 
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long() 
    loss = F.cross_entropy(score, answer) 
    if imitation_loss == True: 
        loss += infer[1]
    
    return loss


