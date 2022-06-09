# fine tunning for imitative mixed image text representation 
import torch  
import os 
from transformers import AdamW
from tqdm import tqdm 

from dataset import CocoConfig, CocoDataLoader
from model import IMITERForImageAndTextRetrieval, compute_image_text_retrieval_loss


def train(model, train_loader, optimizer, train_config, epoch): 
    model.train() 
    iteration = 0 
    loss_cum = 0
    with tqdm(enumerate(train_loader), total=len(train_loader)) as t:
        for idx, batch in t:  
            iteration += 1 
            loss = compute_image_text_retrieval_loss(model, batch, train_config) 
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



def main(): 
    train_config = CocoConfig() 
    data_loader = CocoDataLoader(train_config) 
    train_loader = data_loader.train_dataloader()  

    model = IMITERForImageAndTextRetrieval.from_pretrained(train_config.tokenizer_path) 
    model = model.to(train_config.device) 
    model.train_only_imitation_network()
    
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate) 

    for epoch in range(train_config.max_epoch): 
        train(model, train_loader, optimizer, train_config, epoch) 

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(train_config.ckpt_path, 'imiter/latest.pth'))
        break 


if __name__ == "__main__":
    main()