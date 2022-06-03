import torch 
from tqdm import tqdm 



from dataset import CocoConfig, CocoDataLoader 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def train(train_loader): 
    with tqdm(enumerate(train_loader), total=len(train_loader)) as t:
        for idx, batch in t:  
            print(batch.keys()) 
            break 



def main(): 
    train_config = CocoConfig() 
    dataloader = CocoDataLoader(train_config) 
    train_loader = dataloader.train_dataloader()  



    for epoch in range(train_config.max_epoch): 
        train(train_loader)
        break 


if __name__ == "__main__":
    main()