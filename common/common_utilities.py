import numpy as np


def calc_overlap(dres1, f1, dres2, f2):
    """
        Calculate the overlap between the f1 and elements in f2.
        f2 can be an array, but f1 is scalar.

        returns : this will find the overlap between dres1(f1) (only one) and all detection windows in dres2(f2(:))

        Args:
            dres1: list of image bounding box
            f1: Index into dres1. This should be a scalar value
            dres2: list of image bounding box
            f2: Index into dres2. This is an array.
    """
    # print(type(f2))
    if isinstance(f2, np.int64) or isinstance(f2, int):
        n = 1
    else:
        n = f2.shape[0]

    cx1 = dres1['x'][f1]
    cx2 = dres1['x'][f1] + dres1['w'][f1] - 1
    cy1 = dres1['y'][f1]
    cy2 = dres1['y'][f1] + dres1['h'][f1] - 1

    gx1 = dres2['x'][f2]
    gx2 = dres2['x'][f2] + dres2['w'][f2] - 1
    gy1 = dres2['y'][f2]
    gy2 = dres2['y'][f2] + dres2['h'][f2] - 1

    # area
    ca = dres1['h'][f1] * dres1['w'][f1]
    ga = dres2['h'][f2] * dres2['w'][f2]

    # find the overlapping area
    xx1 = np.maximum(cx1, gx1)
    yy1 = np.maximum(cy1, gy1)
    xx2 = np.minimum(cx2, gx2)
    yy2 = np.minimum(cy2, gy2)
    w = xx2 - xx1 + 1
    h = yy2 - yy1 + 1

    inds = np.where((w > 0) * (h > 0))[0]
    # todo added column dimension, earlier np.zeros((n))
    ov = np.zeros((1, n))
    ov_n1 = np.zeros((1, n))
    ov_n2 = np.zeros((1, n))
    if inds.shape[0] != 0:
        inter = w[inds] * h[inds]  # area of overlap
        u = ca + ga[inds] - w[inds] * h[inds]  # area of union
        ov[inds] = inter / u  # intersection / union
        ov_n1[inds] = inter / ca  # intersection / area in dres1
        ov_n2[inds] = inter / ga[inds]  # intersection / area in dres2

    return ov, ov_n1, ov_n2
