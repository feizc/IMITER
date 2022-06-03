# Dataset Preparation
We utilize several datsets: COCO Captions (COCO), and Flickr 30K Captions (F30K). 

## COCO
https://cocodataset.org/#download

Download [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip) and [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) 

    root
    ├── train2014            
    │   ├── COCO_train2014_000000000009.jpg                
    |   └── ...
    ├── val2014              
    |   ├── COCO_val2014_000000000042.jpg
    |   └── ...          
    └── karpathy
        └── dataset_coco.json 


```python
from dataset import make_coco_arrow
make_coco_arrow(root, arrows_root)
```


## Flickr30K
http://bryanplummer.com/Flickr30kEntities/

Sign [flickr images request form](https://forms.illinois.edu/sec/229675) and download [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

    root
    ├── flickr30k-images            
    │   ├── 1000092795.jpg
    |   └── ...
    └── karpathy
        └── dataset_flickr30k.json

