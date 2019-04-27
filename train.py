import model
import datasets
import losses


from dstorch.detection.icdar2015textloc import ICDAR2015TEXTLOC

import dstorch
import torch
import torchvision

dataset_path = '/srv/hd1/data/ndurasov/datasets/icdar2015/text_localization/'

class Trainer():

    def __init__(self):
        
        # model creation
        self.model = model.east.EAST()
        
        # transform obj for dataset 
        trans = torchvision.transforms.Compose([datasets.augmentations.QUAD(),
                                        datasets.augmentations.RandomCrop((512, 512)),
                                        datasets.augmentations.Normalize()])
        
        # datasets object 
        self.dataset_train = ICDAR2015TEXTLOC(dp, split='train',
                                              transform=trans,
                                              transform_type='dict')
        
        self.dataset_test = ICDAR2015TEXTLOC(dp, split='test',
                                             transform=trans,
                                             transform_type='dict')
        
        # dataloaders objects
        def collate_fn(batch):

            images = list()
            bbox_map = list()
            quad_formatting = list()

            for b in batch:
                images.append(torch.from_numpy(b['image']))
                bbox_map.append(b['bbox_map'])
                quad_formatting.append(b['quad_formatting'])

            images = torch.stack(images, dim=0)

            return {"image": images, 
                    "bbox_map": bbox_map, 
                    "quad_formatting": quad_formatting}

        train_dataloader = torch.utils.data.DataLoader(self.dataset_train, 
                                                       batch_size=16,
                                                       collate_fn=collate_fn,
                                                       shuffle=True)
        
        self.test_dataloader = torch.utils.data.DataLoader(self.dataset_test, 
                                                           batch_size=16,
                                                           collate_fn=collate_fn)

    def train_one_epoch(self):
        
        for batch in self.train_dataloader:
            images, bbox_map, quad_formatting = batch['image'], batch['bbox_map'], batch['quad_formatting'] 
            images = images.permute(0, 3, 1, 2)
            bboxes_predictions = self.model(images)

    def validate_one_epoch(self):
        pass

