import models
import datasets
import losses

from dstorch.detection.icdar2015textloc import ICDAR2015TEXTLOC

import torch
import torchvision

dataset_path = '/srv/hd1/data/ndurasov/datasets/icdar2015/text_localization/'


class Trainer():

    def __init__(self, epochs=10):

        # training configs
        self.epochs = epochs
        
        # models creation
        self.model = models.east.EAST()
        
        # transform obj for dataset 
        trans = torchvision.transforms.Compose([datasets.augmentations.QUAD(),
                                                datasets.augmentations.RandomCrop((512, 512)),
                                                datasets.augmentations.Normalize()])
        
        # datasets object 
        self.dataset_train = ICDAR2015TEXTLOC(dataset_path, split='train',
                                              transform=trans,
                                              transform_type='dict')
        
        self.dataset_test = ICDAR2015TEXTLOC(dataset_path, split='test',
                                             transform=trans,
                                             transform_type='dict')
        
        # dataloaders objects
        self.train_dataloader = torch.utils.data.DataLoader(self.dataset_train,
                                                            batch_size=16,
                                                            shuffle=True)
        
        self.test_dataloader = torch.utils.data.DataLoader(self.dataset_test, 
                                                           batch_size=16)

        # losses
        self.loss_cls = losses.BalancedXent()
        self.loss_geom = losses.QuadGeometry()

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1e-3, amsgrad=True)

    def train_one_epoch(self):

        print("*" * 20, " EPOCH ", "*" * 20)

        for batch in self.train_dataloader:

            images = batch['image'].permute(0, 3, 1, 2)
            bbox_maps = batch['bbox_map'].unsqueeze(1).float()
            quad = batch['quad_formatting'].permute(0, 3, 1, 2)
            ignore_map = batch['ignore_map'].unsqueeze(1)

            prediction = self.model(images)
            l1 = self.loss_cls(prediction['score_map'], bbox_maps,
                          ignore_map.byte())
            l2 = self.loss_geom(prediction['geometry'].float(), quad.float(),
                           bbox_maps.byte())

            if l2 is None:
                final_loss = l1
            else:
                final_loss = l1 + l2

            print("Loss: ", final_loss.item(), l1, l2)

            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

    def validate_one_epoch(self):
        pass


def debug_epoch(epochs=10):
    trans = torchvision.transforms.Compose([datasets.augmentations.QUAD(),
                                            datasets.augmentations.RandomCrop(
                                                (320, 320)),
                                            datasets.augmentations.Normalize()])

    dataset = datasets.debug_dataset.DebugDataset('./debug_data',
                                                  split='train',
                                                  transform=trans)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    model = models.east.EAST()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    loss_cls = losses.BalancedXent()
    loss_geom = losses.QuadGeometry()

    for epoch in range(0, epochs):

        mean_loss = 0

        for batch in dataloader:
            print("*" * 10, " EPOCH {} ".format(epoch), "*" * 10)

            images = batch['image'].permute(0, 3, 1, 2)
            bbox_maps = batch['bbox_map'].unsqueeze(1).float()
            quad = batch['quad_formatting'].permute(0, 3, 1, 2)
            ignore_map = batch['ignore_map'].unsqueeze(1)

            prediction = model(images)
            l1 = loss_cls(prediction['score_map'], bbox_maps,
                          ignore_map.byte())
            l2 = loss_geom(prediction['geometry'].float(), quad.float(),
                           bbox_maps.byte())

            if l2 is None:
                final_loss = l1
            else:
                final_loss = l1 + l2

            mean_loss += final_loss

            print("classification loss: {}".format(l1))
            print("geometry loss: {}".format(l2))
            print("full loss: {}".format(mean_loss / (epoch + 1)))

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()


