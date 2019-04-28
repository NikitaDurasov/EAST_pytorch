from . import dataset_utils
import numpy as np


class RBOX:

    def __init__(self, scale=0.15):
        self.scale = scale

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bbox']

        bbox_map = np.zeros(image.shape[:2])
        dist_map = np.zeros(list(image.shape[:2]) + [2])
        for bbox in bboxes:
            rectangle_bbox = dataset_utils.generate_minimum_quad_orthogon(bbox)
            croped_bbox = dataset_utils.shrink_bbox(rectangle_bbox,
                                                    scale=self.scale)
            bb_interior = dataset_utils.generate_bbox_interion(croped_bbox,
                                                               image.shape[:2])
            distances = dataset_utils.rectangle_borders_distances(croped_bbox,
                                                                  bb_interior)
            bbox_map[bb_interior[:, 1], bb_interior[:, 0]] = 1
            dist_map[bb_interior[:, 1], bb_interior[:, 0]] = distances.T

        return {"image": image,
                "bbox_map": bbox_map,
                "distances_map": dist_map}

class QUAD:

    def __init__(self, scale=0.15):
        self.scale = scale

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bbox']

        bbox_map = np.zeros(image.shape[:2])
        quad_formatting = np.zeros(list(image.shape[:2]) + [8])
        ignore_map = np.ones(image.shape[:2])
        for bbox in bboxes:
            rectangle_bbox = dataset_utils.generate_minimum_quad_orthogon(bbox)
            crop_bbox = dataset_utils.shrink_bbox(rectangle_bbox,
                                                  scale=self.scale)

            bb_interior = dataset_utils.generate_bbox_interion(crop_bbox,
                                                               image.shape[:2])
            full_interior = dataset_utils.generate_bbox_interion(rectangle_bbox,
                                                                 image.shape[:2])

            ignore_map[full_interior[:, 1], full_interior[:, 0]] = 0
            ignore_map[bb_interior[:, 1], bb_interior[:, 0]] = 1
            bbox_map[bb_interior[:, 1], bb_interior[:, 0]] = 1
            quad = dataset_utils.quad_shifts(bbox, bb_interior)
            quad_formatting[bb_interior[:, 1], bb_interior[:, 0]] = quad

        return {"image": image,
                "bbox_map": bbox_map,
                "quad_formatting": quad_formatting,
                "ignore_map": ignore_map}

    
class RandomCrop:
    
    def __init__(self, crop_size=(512, 512)):
        self.crop_size = crop_size 
        
    def __call__(self, sample):
        image = sample['image']
        bbox_map = sample['bbox_map'] 
        quad_formatting = sample['quad_formatting']
        ignore_map = sample['ignore_map']

        
        new_x_b = np.random.randint(0, image.shape[1] - self.crop_size[1])
        new_y_b = np.random.randint(0, image.shape[0] - self.crop_size[0])
        
        new_x_e = new_x_b + self.crop_size[1]
        new_y_e = new_y_b + self.crop_size[0]
            
        new_sample = {"image": image[new_y_b:new_y_e, new_x_b:new_x_e, :],
                      "bbox_map": bbox_map[new_y_b:new_y_e, new_x_b:new_x_e],
                      "quad_formatting": quad_formatting[new_y_b:new_y_e, new_x_b:new_x_e, :],
                      "ignore_map": ignore_map[new_y_b:new_y_e, new_x_b:new_x_e]}
        
        return new_sample
    
class Normalize:
    
    def __init__(self):
        self.imagenet_stats = {'mean': np.array([0.485, 0.456, 0.406]),
                               'std': np.array([0.229, 0.224, 0.225])}
    
    def __call__(self, sample):
        image = sample['image'].astype('float32') / 255
        bbox_map = sample['bbox_map'] 
        quad_formatting = sample['quad_formatting']
        ignore_map = sample['ignore_map']
        
        image -= self.imagenet_stats['mean'].reshape(1, 1, 3)
        image /= self.imagenet_stats['std'].reshape(1, 1, 3)
        
        new_sample = {"image": image,
                      "bbox_map": bbox_map,
                      "quad_formatting": quad_formatting,
                      "ignore_map": ignore_map}
        
        return new_sample
