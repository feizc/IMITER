# training setting for different datasets 
from transformers import BertTokenizer

class CocoConfig: 
    # Image setting 
    train_transform_keys = ["pixelbert_randaug"]  
    val_transform_keys = ["pixelbert"]
    image_size = 384
    patch_size = 32
    draw_false_image = 1 
    image_only = False

    # Text setting
    draw_false_text = 1
    max_text_len = 40 
    mlm_prob = 0.15 
    tokenizer_path = './ckpt/bert' 
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case="uncased") 

    # traning setting
    batch_size = 8
    max_epoch = 10
    warmup_steps = 0.1
    learning_rate = 1e-4 
    data_root = './data' 
    ckpt_path = './ckpt/itr' 



