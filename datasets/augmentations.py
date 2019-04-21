from . import dataset_utils
import numpy as np


class RBOX:

    def __init__(self, scale=0.15):
        self.scale = scale

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']

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

    # TODO FIX ME quad need to calculate bbox shift for points
    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']

        bbox_map = np.zeros(image.shape[:2])
        for bbox in bboxes:
            rectangle_bbox = dataset_utils.generate_minimum_quad_orthogon(bbox)
            crop_bbox = dataset_utils.shrink_bbox(rectangle_bbox,
                                                  scale=self.scale)
            bb_interior = dataset_utils.generate_bbox_interion(crop_bbox,
                                                               image.shape[:2])
            bbox_map[bb_interior[:, 1], bb_interior[:, 0]] = 1

        return {"image": image,
                "bbox_map": bbox_map,
                "bboxes": bboxes}
