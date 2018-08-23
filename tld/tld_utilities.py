import numpy as np
from scipy import spatial


def bb_height(bb):
    """
    This file is part of tld.
    :param bb:
    :return:
    """
    return bb[3] - bb[1] + 1


def bb_width(bb):
    """
    This file is part of tld.
    :param bb:
    :return:
    """
    return bb[2] - bb[0] + 1


def bb_isdef(bb):
    """
    This file is part of tld.
    :param bb:
    :return:
    """
    return np.isfinite(bb[0])


def bb_isout(bb, imsize):
    """
    This file is part of tld.
    :param bb:
    :param imsize:
    :return:
    """
    # todo difference in this and nidhin's code
    idx_out = (bb[0] >= imsize[1]) or (bb[1] >= imsize[0]) or (bb[2] < 1) or (bb[3] < 1)
    return idx_out


def bb_near_border(bb, width, height):
    """
    To check if the bounding box is near image border or not
    :param bb:
    :param width:
    :param height:
    :return:
    """
    # todo write the implementation
    fraction = 0.05
    flag = bb[0] > fraction * width and bb[1] > fraction * height and \
           bb[2] < (1 - fraction) * width and bb[3] < (1 - fraction) * height
    return not flag
    pass


def bb_center(bb):
    """
    returns the coordinates of center
    :param bb:
    :return:
    """
    if len(bb) == 0:
        center = None
        return center
    center = 0.5 * np.array([bb[0] + bb[2], bb[1] + bb[3]])
    return center


def bb_points(BB, num_m, num_n, margin):
    """
    Generates numM x numN points on BBox.
    :param BB:
    :param num_m:
    :param num_n:
    :param margin:
    :return:
    """
    bb = np.zeros(BB.shape)
    bb[0:2] = BB[0:2] + margin
    bb[2:4] = BB[2:4] - margin

    num_stepW = num_n
    num_stepH = num_m

    if num_m == 1 and num_n == 1:
        pt = bb_center(bb)
        return pt

    if num_m == 1 and num_n > 1:
        c = bb_center(bb)
        pt = np.array(np.meshgrid(np.linspace(bb[0], bb[2], num_stepW), np.array([c[1]]))).T.reshape(-1, 2).T
        return pt

    if num_m > 1 and num_n == 1:
        c = bb_center(bb)
        pt = np.array(np.meshgrid(np.array([c[0]]), np.linspace(bb[1], bb[3], num_stepH))).T.reshape(-1, 2).T
        return pt

    pt = np.array(np.meshgrid(np.linspace(bb[0], bb[2], num_stepW), np.linspace(bb[1], bb[3], num_stepH))).T.reshape(-1,
                                                                                                                     2).T

    if pt.shape[1] < num_m * num_n:
        count = num_m * num_n - pt.shape[1]
        app = np.repeat(pt[:, -1], count, 0)
        pt = np.array([pt, app])
    return pt


# todo looks fine
def bb_predict(BB0, pt0, pt1):
    """

    :param BB0:
    :param pt0:
    :param pt1:
    :return:
    """
    of = pt1 - pt0
    dx = np.median(of[0, :])
    dy = np.median(of[1, :])

    d1 = spatial.distance.pdist(pt0.T, 'euclidean')
    d2 = spatial.distance.pdist(pt1.T, 'euclidean')
    s = np.median(d2 / d1)

    s1 = 0.5 * (s - 1) * bb_width(BB0)
    s2 = 0.5 * (s - 1) * bb_height(BB0)

    BB1 = np.array([BB0[0] - s1, BB0[1] - s2, BB0[2] + s1, BB0[3] + s2]) + [dx, dy, dx, dy]

    return BB1, [s1, s2]


# todo refactor this
def bb_rescale_relative(BB, s):
    """

    :param BB:
    :param s:
    :return:
    """
    if len(BB) == 0:
        return None

    BB = BB[0:4]

    if len(s) == 1:
        s = s * [1, 1]

    s1 = 0.5 * (s[0] - 1) * bb_width(BB)
    s2 = 0.5 * (s[1] - 1) * bb_height(BB)
    BB = BB + [-s1, -s2, s1, s2]
    return BB


# todo refactor this
def bb_shift_relative(bb, shift):
    """

    :param bb:
    :param shift:
    :return:
    """
    if bb.size == 0:
        return
    bb_shift = np.zeros(4, dtype=np.float64)

    bb[0] = bb[0] + bb_width(bb) * shift[0]
    bb[1] = bb[1] + bb_height(bb) * shift[1]
    bb[2] = bb[2] + bb_width(bb) * shift[0]
    bb[3] = bb[3] + bb_height(bb) * shift[1]
    return bb


def bb_shift_absolute(bb, shift):
    """

    :param bb:
    :param shift:
    :return:
    """
    bb[0] = bb[0] + shift[0]
    bb[1] = bb[1] + shift[1]
    bb[2] = bb[2] + shift[0]
    bb[3] = bb[3] + shift[1]
    return bb


# todo implement if needed
def bb_union(BB1, BB2):
    """

    :param BB1: bounding box 1
    :param BB2: bounding box 2
    :return:
    """
    pass
