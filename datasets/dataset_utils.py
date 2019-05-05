from cv2 import minAreaRect, boxPoints
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw

def generate_minimum_quad_orthogon(quad_bbox):
    """Generate rectangle with minimal square that surrounds given polygon.

    Args:
        quad_bbox (numpy.array): float numpy array with shape (8,) that
        contains coordinates of polygon vertexes.

    Returns:
        numpy.array: float numpy array with shape (8,) that contains
        coordinates of minimal square rectangle.

    """
    points_pairs = quad_bbox.reshape(-2, 2)
    rect = minAreaRect(points_pairs)
    box = boxPoints(rect).ravel()
    return box


def read_gt(filename):
    lines = open(filename).readlines()
    bboxes = np.array(list(map(lambda x: x.split(',')[:-1], lines)))
    return bboxes.astype('float32')


def draw_bbox(bbox, c='r'):
    """Drawing given bbox borders.

    Args:
        bbox (numpy.array): float numpy array with shape (8,) that
        contains coordinates of polygon vertexes.

    """
    plt.plot(bbox[::2], bbox[1::2], c=c)
    plt.plot([bbox[-2], bbox[0]],
             [bbox[-1], bbox[1]], c=c)


def draw_bboxes(img, bboxes, c='r'):
    """Drawing given image and bboxes borders

    Args:
        img (numpy.array): float numpy array of image with shape (*, *, 3).

        bboxes (numpy.array): float numpy array with shape (*, 8) that
        contains coordinates of polygons vertexes.

    """
    plt.imshow(img)
    for bbox in bboxes:
        draw_bbox(bbox, c)


def normalize(vector):
    return vector / np.linalg.norm(vector)


def shrink_bbox(bbox, scale=0.2):
    """Clip original rectangle from both size with scale * current_side_length.
    Result of this operation is rectangle with smaller square nested in
    original bbox.

    Examples:
        >>> bbox = np.array([-2, -2, -2, 2, 2, 2, 2, -2])
        >>> clipped_bbox = shrink_bbox(bbox, 0.25)
        >>> print(clipped_bbox)
        array([-1, -1, -1,  1,  1,  1,  1, -1])

    Args:
        bbox (numpy.array): float numpy array with shape (8,) that
        contains coordinates of polygon vertexes.

        scale (float): reduction measure of rectangle sides, e.g. scale=0.25
        rectangle sides will be reduced twice, scale=0.5 - all vertices are
        tightened to one point.

    Returns:
        numpy.array: float numpy array with shape (8,) that contains
        coordinates of reduced bbox

    """
    bbox_points = bbox.reshape(4, 2)

    first_direction = scale * (bbox_points[1] - bbox_points[0])
    second_direction = scale * (bbox_points[2] - bbox_points[1])

    new_points = bbox_points.copy()

    new_points[0] = new_points[0] + first_direction
    new_points[3] = new_points[3] + first_direction
    new_points[2] = new_points[2] - first_direction
    new_points[1] = new_points[1] - first_direction

    new_points[0] = new_points[0] + second_direction
    new_points[3] = new_points[3] - second_direction
    new_points[2] = new_points[2] - second_direction
    new_points[1] = new_points[1] + second_direction

    return bbox_order(new_points.ravel())


def rectangle_borders_distances(bbox, points):
    """Calculate distances from points to borders of bbox. This function
    assumes that first point in bbox is upper left one and other point
    follows in clockwise order.

    Args:
        bbox (numpy.array): float numpy array with shape (8,) that
        contains coordinates of polygon vertexes.

        points (numpy.array): float numpy array with shape (*, 8) that
        contains coordinates of considered points.

    Returns:
        numpy.array: float numpy array with shape (*, 2) that contains
        distances from corresponding point to bbox borders.

    """
    bbox_points = bbox.reshape(4, 2)
    a, b = bbox_points[0], bbox_points[2]
    a_p = a - points
    b_p = b - points

    first_direction = normalize(bbox_points[1] - bbox_points[0])
    second_direction = normalize(bbox_points[2] - bbox_points[1])

    h_dist = np.vstack([np.abs(np.dot(a_p, first_direction)),
                        np.abs(np.dot(b_p, first_direction))])

    v_dist = np.vstack([np.abs(np.dot(a_p, second_direction)),
                        np.abs(np.dot(b_p, second_direction))])

    return np.vstack([np.min(h_dist, axis=0),
                      np.min(v_dist, axis=0)])


def generate_bbox_interion(bbox, shape):
    """Generates coordinates of bbox interior points

    Args:
        bbox (numpy.array): float numpy array with shape (8,) that
        contains coordinates of polygon vertexes.

        shape (tuple): two elements tuple with height and width of image

    Returns:
        numpy.array: float numpy array with shape (*, 2) that contains
        coordinates of bbox interior points

    """
    shape = shape[1::-1]
    points = skimage.draw.polygon(bbox[::2], bbox[1::2], shape=shape)
    return np.vstack([points[0], points[1]]).T


def quad_shifts(bbox, points):
    """Calculates vector shift for points to bbox vertexes

    Args:
        bbox (numpy.array): float numpy array with shape (8,) that
        contains coordinates of polygon vertexes.

        points (numpy.array): float numpy array with shape (*, 8) that
        contains coordinates of considered points.

    Returns:
        numpy.array: float numpy array with shape (*, 8) that contains
        shifts vectors from points to bbox vertexes, e.g.
        bbox = (v1, v2, v3, v4) and point in z = (x, y), then result is
        (v1 - x, v2 - x, v3 - x, v4 - x)

    """
    upper_left_diff = bbox.reshape(4, 2)[0] - points
    upper_right_diff = bbox.reshape(4, 2)[1] - points
    lower_right_diff = bbox.reshape(4, 2)[2] - points
    lower_left_diff = bbox.reshape(4, 2)[3] - points
    return np.hstack([upper_left_diff, upper_right_diff,
                      lower_right_diff, lower_left_diff])


def bbox_order(bbox):
    """Fixing bbox vertexes order: upper-left points is the first, other
    follows in clockwise manner.

    Args:
        bbox (numpy.array): float numpy array with shape (8,) that
        contains coordinates of polygon vertexes.

    Returns:
        numpy.array: float numpy array with shape (8,) that
        contains coordinates of fixed order polygon vertexes.

    """

    def mapping(x):
        if x == 0:
            return 3
        elif x == 1:
            return 0
        elif x == 2:
            return 2
        elif x == 3:
            return 1

    mapping = np.vectorize(mapping)

    box_points = bbox.reshape(4, 2)
    mean_point = box_points.mean(axis=0)
    new_res = quad_shifts(bbox, mean_point).reshape(4,2)
    positive_values = new_res >= 0
    new_order = mapping(positive_values.sum(axis=1) + positive_values[:, 0])
    return box_points[new_order].ravel()


def short_length_bbox_size(bboxes):
    bb_points = bboxes.reshape(-1, 4, 2)
    first_lengths = np.linalg.norm(bb_points[:, 1] - bb_points[:, 0], axis=1)
    second_lengths = np.linalg.norm(bb_points[:, 2] - bb_points[:, 1], axis=1)
    return np.stack([first_lengths, second_lengths]).T.min(axis=1)
