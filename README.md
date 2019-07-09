# EAST_pytorch

Pytorch implementation of EAST detector for text -- this version is not based on original TensorFlow authors code.

## Usage 

First of all you need to install [Pytorch](https://pytorch.org/?utm_source=Google&utm_medium=PaidSearch&utm_campaign=%2A%2ALP+-+TM+-+General+-+HV+-+RU&utm_adgroup=Install+PyTorch&utm_keyword=%2Binstall%20%2Bpytorch&utm_offering=AI&utm_Product=PyTorch&gclid=Cj0KCQjw6cHoBRDdARIsADiTTzbpH_VFIFaOoEmjySWPiLx9J5wkLwud2-SnaUIDQtpTXDNL1qEadcAaAlFREALw_wcB) and [dstorch](https://github.com/nikitadurasov/dstorch)

Other required libraries: 
* albumentations
* torchvision

For training you need to download ICDAR 2015:

*Note: you should change the gt text file of icdar2015's filename to img_*.txt instead of gt_img_*.txt*

To start training procedure you need to run 

```python
python train.py
```
Please find useful function for bounding box processing in [datasets_utils](https://github.com/nikitadurasov/EAST_pytorch/blob/master/datasets/dataset_utils.py) and basic implementation of EAST in models. Lanms directory consist of original authors code for locally aware NMS algorithm from paper (requires at least gcc 6 ).
