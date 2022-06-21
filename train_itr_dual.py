# fine tunning for universal image text representation with dual encoder 


import torch  
import os 
import functools 
from tqdm import tqdm 

from dataset import CocoConfig, CocoDataLoader
from model import ClipModel, ClipConfig


def train(model, train_loader, optimizer, train_config, epoch): 
    model.train() 
    iteration = 0 
    loss_cum = 0
    with tqdm(enumerate(train_loader), total=len(train_loader)) as t:
        for idx, batch in t:  
            iteration += 1 
            loss = model(
                pixel_values = batch['image'][0].to(train_config.device),
                input_ids = batch['text_ids'].to(train_config.device),
                attention_mask = batch['text_masks'].to(train_config.device),
            )[0]
            
            loss = loss / train_config.gradient_accumulation_steps 
            loss.backward()
            if iteration % train_config.gradient_accumulation_steps == 0: 
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_norm) 
                optimizer.step() 
                optimizer.zero_grad() 
            loss_cum += loss.item() 
            t.set_description('Epoch %i' % epoch)
            t.set_postfix(loss=loss_cum / (idx+1))
            break


# evaluate performance with dual encoder form 
def eval(model, data_loader, train_config): 
    text_dataset = data_loader.make_no_false_val_dataset() 
    tokenizer = train_config.tokenizer 
    text_loader = torch.utils.data.DataLoader(
        text_dataset, 
        batch_size=16, 
        pin_memory=True, 
        collate_fn=functools.partial(
            data_loader.train_dataset.collate,
            mlm_collator=data_loader.mlm_collator,
        ),
    ) 

    image_dataset = data_loader.make_no_false_val_dataset(image_only=True) 
    image_loader = torch.utils.data.DataLoader(
        image_dataset, 
        batch_size=1, 
        pin_memory=True, 
        collate_fn=functools.partial(
            data_loader.train_dataset.collate,
            mlm_collator=data_loader.mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm(text_loader, desc="text prefetch loop"): 
        with torch.no_grad():
            text_features = model.step(
                input_ids = _b["text_ids"].to(train_config.device),
                attention_mask = _b["text_masks"].to(train_config.device),
            )
        text_preload.append(
            {
                "text_features": text_features,
                "img_index": _b["img_index"],
            }
        ) 
        break 

    tiids = list()
    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids) 

    rank_scores = list() 
    rank_iids = list() 

    for _b in tqdm(image_loader, desc='rank loop'): 
        pixel_values = _b['image'][0].to(train_config.device)
        with torch.no_grad(): 
            image_features = model.step(
                pixel_values = pixel_values
            ) 


        image_batch_score = list()
        for text_batch in text_preload: 
            score = cos_similarity(image_features, text_batch['text_features']) 
            image_batch_score.append(score) 

        image_batch_score = torch.cat(image_batch_score) 
        rank_scores.append(image_batch_score.cpu().tolist())
        rank_iids.append(_b['img_index'][0])
        
        break 

    iids = torch.tensor(rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(rank_scores)
    scores = scores.view(len(iids), -1) 

    topk1 = scores.topk(1, dim=1)
    topk1_iids = tiids[topk1.indices]
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()
    topk1 = scores.topk(1, dim=0)
    topk1_iids = iids[topk1.indices]
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()
    output = (ir_r1, tr_r1)


    if train_config.report_all_merics == True:

        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk10_iids = tiids[topk10.indices]
        topk5_iids = tiids[topk5.indices]
        tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()

        topk10 = scores.topk(10, dim=0)
        topk5 = scores.topk(5, dim=0)
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
        ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()

        output += (ir_r5, ir_r10, tr_r5, tr_r10)
    print(output)
    return output 



def main(): 
    train_config = CocoConfig() 
    data_loader = CocoDataLoader(train_config) 
    train_loader = data_loader.train_dataloader()  

    model_config = ClipConfig()
    model = ClipModel(model_config)
    model = model.to(train_config.device) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate) 

    for epoch in range(train_config.max_epoch): 
        train(model, train_loader, optimizer, train_config, epoch) 
        eval(model, data_loader, train_config)

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(train_config.ckpt_path, 'clip/latest.pth'))
        break 


if __name__ == "__main__":
    main()