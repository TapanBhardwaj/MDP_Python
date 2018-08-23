from tld.tld_utilities import *
import numpy as np
import cv2


def median2(x):
    # Median without nan

    y = x[np.isfinite(x)]
    m = np.median(y)
    return m

def compute_velocity(tracker):
    '''
    Calculate the velocity from last 3-frames if exists
    :param tracker:
    :return: array which contain 2 elements
    '''
    fr = np.unique(tracker.frame_ids).astype(np.dtype('d'))
    num = len(fr)

    # only use the past 3 frames
    if num > 3:
        fr = fr[num - 3:num]
        num = 3

    # compute centers
    centers = np.zeros((num, 2))
    for i in range(num):
        index = np.where(tracker.frame_ids == fr[i])[0]
        for j in range(len(index)):
            ind = index[j]
            c = [(tracker.x1[ind] + tracker.x2[ind]) / 2, (tracker.y1[ind] + tracker.y2[ind]) / 2]
            centers[i, :] = centers[i, :] + c
        if len(index):
            centers[i, :] = centers[i, :] / len(index)

    count = 0
    vx = 0
    vy = 0
    cx = centers[:, 0]
    cy = centers[:, 1]
    for j in range(1, num):
        vx = vx + (cx[j] - cx[j - 1]) / (fr[j] - fr[j - 1])
        vy = vy + (cy[j] - cy[j - 1]) / (fr[j] - fr[j - 1])
        count += 1

    if count:
        vx = vx / count
        vy = vy / count
    v = np.array([vx, vy])
    return v

def lk_cv(i, I, J, xFI, xFII, level=5):
    """lk optical flow implementation based on opencv

    Arguments:
        i {[type]} -- Not used
        I {ndarray} -- current image
        J {ndarray} -- next image
        xFI {[type]} -- Points(100) in I. Predictions for points in J are made using this points.
        xFII {[type]} -- Points(100) in J predeicted using motion flow.

    Keyword Arguments:
        level {int} -- [description] (default: {5})

    Returns:
        [type] -- contains the predicted points in I and J, euclideanDistance, normCrossCorrelation
    """

    MAX_COUNT = 500
    MAX_IMG = 2
    win_size = 4
    Winsize = 10
    imageSize = (I.shape[0], I.shape[1])
    nPts = xFI.shape[1]
    xFI = xFI.T.reshape(-1, 1, 2).astype(np.float32)
    xFII = xFII.T.reshape(-1, 1, 2).astype(np.float32)
    points = [np.copy(xFI), np.copy(xFII), np.copy(xFI)]

    # todo change this to support flownet and unflow
    points[1], status, _err = cv2.calcOpticalFlowPyrLK(I, J, points[0], points[1], winSize=(win_size, win_size),
                                                       maxLevel=level, criteria=(
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03), flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

    points[2], status, _err = cv2.calcOpticalFlowPyrLK(J, I, points[1], points[2], winSize=(win_size, win_size),
                                                       maxLevel=level, criteria=(
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03), flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

    points = [points[i].reshape(-1, 2) for i in range(3)]

    # normCrossCorrelation
    ncc = np.zeros((nPts, 1))
    for i in range(nPts):
        if status[i] == 1:
            rec0 = cv2.getRectSubPix(I, patchSize=(Winsize, Winsize), center=(points[0][i][0], points[0][i][1]))
            rec1 = cv2.getRectSubPix(J, patchSize=(Winsize, Winsize), center=(points[1][i][0], points[1][i][1]))
            res = cv2.matchTemplate(rec0, rec1, method=cv2.TM_CCOEFF_NORMED)
            ncc[i][0] = res[0][0]

    # euclideanDistance
    fb = np.linalg.norm(points[0] - points[2], axis=1).reshape(-1, 1)

    xFj = np.concatenate((points[1], fb, ncc), axis=1)
    xFj[(status != 1).reshape(-1)] = [np.nan, np.nan, np.nan, np.nan]

    return xFj.T


def LK(I, J, BB1, BB2, margin, level):
    """Estimates motion from bounding box BB1 in frame I to bounding box BB2 in frame J
        obj is the background model

    Arguments:
        I  -- current image
        J  -- next image
        BB1  -- Bounding box in I
        BB2  -- Bounding box in J
        margin  -- gap for getting the 100 points(10x10 grid margin)
        level  -- level for pyramidOptical flow

    Returns:
        Bounding box, flags, medFB -- optical flow results.
    """

    # initialize output variables
    BB3 = None  # estimated bounding

    # exit function if BB1 or BB2 is not defined
    if len(BB1) == 0 or not bb_isdef(BB1):
        return

    # estimate BB3
    xFI = bb_points(BB1, 10, 10, [margin[0], margin[1]])  # generate 10x10 grid of points within BB1
    if len(BB2) == 0 or not bb_isdef(BB2):
        xFII = xFI
    else:
        xFII = bb_points(BB2, 10, 10, [margin[0], margin[1]])

    # track all points by Lucas-Kanade tracker from frame I to frame J,
    # estimate Forward-Backward error, and NCC for each point
    xFJ = lk_cv(2, I, J, xFI, xFII, level)

    medFB = median2(xFJ[2, :])  # get median of Forward-Backward error
    medNCC = median2(xFJ[3, :])  # get median for NCC
    idxF = np.logical_and(xFJ[2, :] <= medFB, xFJ[3, :] >= medNCC)  # get indexes of reliable points
    BB3, _ = bb_predict(BB1, xFI[:, idxF], xFJ[0:2, idxF]);  # estimate BB2 using the reliable points only

    index = xFI[0, :] < (BB1[0] + BB1[2]) / 2
    medFB_left = median2(xFJ[2, index])

    index = xFI[0, :] >= (BB1[0] + BB1[2]) / 2
    medFB_right = median2(xFJ[2, index])

    index = xFI[1, :] < (BB1[1] + BB1[3]) / 2
    medFB_up = median2(xFJ[2, index])

    index = xFI[1, :] >= (BB1[1] + BB1[3]) / 2
    medFB_down = median2(xFJ[2, index])

    # save selected points (only for display purposes)
    xFJ = xFJ[:, idxF]

    flag = 1
    # detect failures
    # bounding box out of image
    if ~bb_isdef(BB3) or bb_isout(BB3, J.shape):
        flag = 2
        return [BB3, xFJ, flag, medFB, medNCC, medFB_left, medFB_right, medFB_up, medFB_down]

    # too unstable predictions
    if medFB > 10:
        flag = 3
        return [BB3, xFJ, flag, medFB, medNCC, medFB_left, medFB_right, medFB_up, medFB_down]

    return [BB3, xFJ, flag, medFB, medNCC, medFB_left, medFB_right, medFB_up, medFB_down]


