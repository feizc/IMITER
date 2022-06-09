from model import ITRForImageAndTextRetrieval, ITRConfig, keys_to_transforms
from transformers import BertTokenizer

import requests
from PIL import Image


def test_itr(image, tests):
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





if __name__ == "__main__": 
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

    test_itr(image, texts) 
    