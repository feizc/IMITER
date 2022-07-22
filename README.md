# IMITER

This is the PyTorch implementation for inference and training of the image-text models as described in: 
> **IMITER: Imitative Mixed Image-TExt Representation learning**

We target to explore the function of sinle encoder, dual encoder, and mixed encoder for image-text representation learning. 
This repository supports finetuning IMITER on
[NLVR2](http://lil.nlp.cornell.edu/nlvr/), [VQA](https://visualqa.org/), [VCR](https://visualcommonsense.com/),
[SNLI-VE](https://github.com/necla-ml/SNLI-VE), 
Image-Text Retrieval for [COCO](https://cocodataset.org/#home) and
[Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/), and
[Referring Expression Comprehensions](https://github.com/lichengunc/refer) (RefCOCO, RefCOCO+, and RefCOCO-g).
Both IMITER-base and IMITER-large pre-trained checkpoints are released.


- [x] imitation for imgae/text key/value matrix 
- [x] shared image/text embedding with linear projection
- [x] mask perturbation consistency 



## 1 Requirements 

We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards. 


## 2 Quick Start 

We use image-text retrival in COCO dataset as an end-to-end example for using this code base.


## 3 Dataset Preparation 
See [`Dataset.md`](dataset/README.md)



