# fine tunning for imitative mixed image text representation with dual encoder 
import torch  
import os 
import functools 
from tqdm import tqdm 
import torch.nn as nn 
import torch.nn.functional as F 
from einops import rearrange 


from model import ViltForImageAndTextRetrieval, ViltConfig 

from dataset import CocoConfig, CocoDataLoader 

cos_flag = True 

def compute_image_text_retrieval_loss(model, batch, train_config): 
    device = train_config.device 
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
        pixel_values = rearrange(images, "bs fs c h w -> (bs fs) c h w").to(device),
        input_ids = rearrange(text_ids, "bs fs tl -> (bs fs) tl").to(device),
        attention_mask = rearrange(text_masks, "bs fs tl -> (bs fs) tl").to(device),
    ) 
    if cos_flag == True: 
        score = infer[0] 
    else:
        score = infer.logits[:, 0] 
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)  # (bsz, false_len + 1)
    answer = torch.zeros(_bs).to(score).long() #(bsz, ) 

    index = torch.argmax(score, dim=1) 
    acc = torch.eq(index.cpu(), answer.cpu()).float()
    acc = acc.sum() / _bs 
    loss = F.cross_entropy(score, answer) 
    
    return loss, acc 



def train(model, train_loader, optimizer, train_config, epoch): 
    model.train() 
    iteration = 0 
    loss_cum = 0
    acc_cum = 0
    with tqdm(enumerate(train_loader), total=len(train_loader)) as t:
        for idx, batch in t:  
            iteration += 1 
            
            loss, acc = compute_image_text_retrieval_loss(model, batch, train_config) 
            loss = loss / train_config.gradient_accumulation_steps  

            loss.backward()
            if iteration % train_config.gradient_accumulation_steps == 0: 
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_norm) 
                optimizer.step() 
                optimizer.zero_grad() 
            loss_cum += loss.item()
            acc_cum += acc.item()

            t.set_description('Epoch %i' % epoch)
            t.set_postfix(loss=loss_cum / (idx+1), acc=acc_cum/(idx+1))
            break



def eval(model, eval_loader, train_config, epoch): 
    iteration = 0 
    loss_cum = 0
    acc_cum = 0
    with tqdm(enumerate(eval_loader), total=len(eval_loader)) as t:
        for idx, batch in t:  
            iteration += 1 
            with torch.no_grad(): 
                loss, acc = compute_image_text_retrieval_loss(model, batch, train_config) 
            
            loss_cum += loss.item()
            acc_cum += acc.item()
            t.set_description('Epoch %i' % epoch)
            t.set_postfix(loss=loss_cum / (idx+1), acc=acc_cum/(idx+1))
            break





def main(): 
    train_config = CocoConfig() 
    data_loader = CocoDataLoader(train_config) 
    train_loader = data_loader.train_dataloader()  
    eval_loader = data_loader.val_dataloader()

    model_config = ViltConfig.from_pretrained('./ckpt/vilt')
    model = ViltForImageAndTextRetrieval(model_config)
    if train_config.resume_flag == True: 
        model.load_state_dict(torch.load('./ckpt/imiter/latest.pth')['state_dict'])
    model = model.to(train_config.device) 

    if train_config.train_only_imitation == True: 
        model.train_only_imitation_network()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate) 
    

    for epoch in range(train_config.max_epoch): 
        train(model, train_loader, optimizer, train_config, epoch) 
        eval(model, eval_loader, train_config, epoch) 
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(train_config.ckpt_path, 'latest.pth'))
        break 


if __name__ == "__main__":
    main()