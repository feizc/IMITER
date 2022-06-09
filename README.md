# IMITER

This is the PyTorch implementation for inference and training of the image-text models as described in: 
> **IMITER: Imitative Mixed Image-TExt Representation learning**

This repository supports finetuning IMITER on
[NLVR2](http://lil.nlp.cornell.edu/nlvr/), [VQA](https://visualqa.org/), [VCR](https://visualcommonsense.com/),
[SNLI-VE](https://github.com/necla-ml/SNLI-VE), 
Image-Text Retrieval for [COCO](https://cocodataset.org/#home) and
[Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/), and
[Referring Expression Comprehensions](https://github.com/lichengunc/refer) (RefCOCO, RefCOCO+, and RefCOCO-g).
Both IMITER-base and IMITER-large pre-trained checkpoints are released.


## Requirements 

We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards. 


## Quick Start 

We use COCO as an end-to-end example for using this code base.


## Dataset Preparation 
See [`Dataset.md`](dataset/README.md)



