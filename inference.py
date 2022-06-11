from lib2to3.pgen2 import token
import torch 
import requests
from PIL import Image
import torch.nn.functional as F 

from model import ITRForImageAndTextRetrieval, ITRConfig, keys_to_transforms, IMITERForImageAndTextRetrieval, IMITERConfig
from transformers import BertTokenizer


def test_itr(image, texts): 
    ckpt_path = './ckpt/bert'
    tokenizer = BertTokenizer.from_pretrained(ckpt_path, do_lower_case="uncased") 

    model = ITRForImageAndTextRetrieval.from_pretrained(ckpt_path)  
    train_transform_keys = ["pixelbert_randaug"]
    transform = keys_to_transforms(train_transform_keys)
    image_tensor = [tr(image) for tr in transform]
    data ={"pixel_values": image_tensor[0].unsqueeze(0)}

    scores = dict()
    for text in texts:
        input_ids = tokenizer(text=text, add_special_tokens=True, return_tensors='pt') 
        input_ids.update(data)
        outputs = model(**input_ids) 
        scores[text] = outputs[0].item() 
    print(scores)


# single encoder for image text representation 
def test_imiter_combine(image, texts): 
    bert_path = './ckpt/bert' 
    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case='uncased')  
    model_config = IMITERConfig()
    param_dict = torch.load('ckpt/imiter/latest.pth')
    model = IMITERForImageAndTextRetrieval(model_config)
    model.load_state_dict(param_dict['state_dict'])
    train_transform_keys = ["pixelbert_randaug"]
    transform = keys_to_transforms(train_transform_keys)
    image_tensor = [tr(image) for tr in transform]
    data ={"pixel_values": image_tensor[0].unsqueeze(0)}

    scores = dict()
    for text in texts:
        input_dict = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=40,
        )
        
        data['input_ids'] = torch.Tensor(input_dict['input_ids']).long().unsqueeze(0) 

        data['attention_mask'] = torch.Tensor(input_dict['attention_mask']).long().unsqueeze(0)
        outputs = model(**data) 
        scores[text] = outputs[0].item() 
    print(scores)







if __name__ == "__main__": 
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

    test_itr(image, texts) 
    # test_imiter_combine(image, texts) 
