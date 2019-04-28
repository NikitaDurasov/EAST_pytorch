from . import east


class BoxesPredictor:

    def __init__(self, thr=0.5):
        self.thr = thr

    def __call__(self, prediction):

        score_maps = prediction["score_map"]
        geometries = prediction["geometry"]
        batch_boxes = []

        for i in range(len(score_maps)):
            score_map = score_maps[i].squeeze()
            geometry = geometries[i].squeeze().permute(1, 2, 0)

            mask = score_map > self.thr
            boxes = geometry[mask]

            batch_boxes.append(boxes)

        return batch_boxes



