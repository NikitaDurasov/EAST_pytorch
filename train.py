import models
import datasets
import losses
import numpy as np
import lanms
import metrics

from dstorch.detection.icdar2015textloc import ICDAR2015TEXTLOC

import torch
import torchvision
import sys

dataset_path = '/srv/hd1/data/ndurasov/datasets/icdar2015/text_localization/'


class Trainer():

    def __init__(self, checkpoint=None):
        
        # models creation
        self.model = models.east.EAST()
        
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))
        
        # transform obj for dataset 
        trans = torchvision.transforms.Compose([datasets.augmentations.QUAD(0.3),
                                                datasets.augmentations.RandomCrop((640, 704)),
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
                                                            batch_size=3,
                                                            shuffle=True)
        
        self.test_dataloader = torch.utils.data.DataLoader(self.dataset_test, 
                                                           batch_size=1)

        # losses
        self.loss_cls = losses.BalancedXent()
        self.loss_geom = losses.QuadGeometry()

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1e-5, amsgrad=True)
        
    def print_train_results(self, epoch, batch, mean_loss, loss, cls_loss):
        # print current accuracy
        sys.stdout.write("\r Training: epoch = {0} ".format(epoch))
        sys.stdout.write("batch: {0}/{1} ".format(batch, len(self.train_dataloader)))
        sys.stdout.write("mean loss = %.3f " % mean_loss)
        sys.stdout.write("loss = %.3f " % loss)
        sys.stdout.write("classification loss = %.3f " % cls_loss)
        sys.stdout.flush()

    def train_one_epoch(self, epoch):
        self.model = self.model.cuda()
        
        mean_loss = 0

        for i, batch in enumerate(self.train_dataloader):
            images = batch['image'].permute(0, 3, 1, 2).cuda()
            bbox_maps = batch['bbox_map'].unsqueeze(1).float().cuda()
            quad = batch['quad_formatting'].permute(0, 3, 1, 2).cuda()
            ignore_map = batch['ignore_map'].unsqueeze(1).cuda()

            prediction = self.model(images)
            l1 = 5e3 * self.loss_cls(prediction['score_map'], bbox_maps,
                          ignore_map.byte())
            l2 = self.loss_geom(prediction['geometry'].float(), quad.float(),
                           bbox_maps.byte())

            if l2 is None:
                final_loss = l1
            else:
                final_loss = l1 + l2
                
            mean_loss += final_loss.item()

            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()
            
            self.print_train_results(epoch, i, mean_loss / (i + 1), 
                                     final_loss.item(),
                                     l1.item())
        
        print()
        self.model = self.model.cpu()
        
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def print_valid_results(self, epoch, batch, recall, precision, iou):
        # print current accuracy
        sys.stdout.write("\r Validation: epoch = {0} ".format(epoch))
        sys.stdout.write("batch: {0}/{1} ".format(batch, len(self.test_dataloader)))
        sys.stdout.write("recall = %.3f " % recall)
        sys.stdout.write("precision = %.3f " % precision)
        sys.stdout.write("IOU = %.3f " % iou)
        sys.stdout.flush()

    def validate_one_epoch(self, epoch):
        
        box_predictor = models.BoxesPredictor(0.9)
        recall_cum = 0
        prec_cum = 0
        iou_cum = 0
        
        self.model = self.model.cuda()

        i = 0
        for _, batch in enumerate(self.test_dataloader):
            images = batch['image'].permute(0, 3, 1, 2).cuda()
            bbox_maps = batch['bbox_map'].unsqueeze(1).float()
            quad = batch['quad_formatting'].permute(0, 3, 1, 2)
            ignore_map = batch['ignore_map'].unsqueeze(1)

            prediction = self.model(images)
            for key in prediction:
                prediction[key] = prediction[key].cpu()
            boxes = box_predictor(prediction)[0]
            boxes_p = np.hstack([boxes.detach().numpy(), np.ones([boxes.shape[0], 1])])
            boxes_pr = lanms.merge_quadrangle_n9(boxes_p, 0.1)
            valid_boxes = boxes_pr[:, :8]
            
            ground_boxes = box_predictor({"score_map": bbox_maps, "geometry": quad.float()})[0].numpy()
            if ground_boxes.shape[0] != 0:
                ground_boxes = np.hstack([ground_boxes, np.ones([ground_boxes.shape[0], 1])])
                ground_boxes = lanms.merge_quadrangle_n9(ground_boxes, 0.1)
                ground_boxes = ground_boxes[:, :8]
                
                recall, prec, iou = metrics.matchBoxes(ground_boxes, valid_boxes, thr=0.5)
                recall_cum += recall
                prec_cum += prec
                iou_cum += iou
                
                self.print_valid_results(epoch, i, 
                                         recall_cum / (i + 1), 
                                         prec_cum / (i + 1),
                                         iou_cum / (i + 1))
                i += 1
        
        print()
        self.model = self.model.cpu()

if __name__ == '__main__':
    
    if sys.argv[1] == 'train':
        trainer = Trainer('./experiments/29_04_1_epoch_12.pth')
        for epoch in range(40):
            trainer.validate_one_epoch(epoch)
            trainer.train_one_epoch(epoch)
            if epoch % 3 == 0:
                trainer.save_model('experiments/30_04_2_epoch_{}.pth'.format(epoch))
    elif sys.argv[1] == 'test':
        trainer = Trainer('./experiments/29_04_1_epoch_12.pth')
        trainer.validate_one_epoch(0)
        

