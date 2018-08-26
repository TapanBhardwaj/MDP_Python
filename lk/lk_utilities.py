from tld.tld_utilities import *
import numpy as np
from svmutil import *
from scipy.misc import imresize
import cv2
import array
from resample import imResample


def median2(x):
    # Median without nan

    y = x[np.isfinite(x)]
    m = np.median(y)
    return m


def distance(u, v):
    """
    **************************************************************************
    Directly taken from Nidhin's code
    **************************************************************************
        assume `u` is of size Mx1
    """
    if u.shape[1] > 1:
        print("Error: distance - u shape mis-match. Need to iterate through second axis")
    u = u.reshape(-1)
    if len(v.shape) == 1:
        num = 1
    else:
        num = v.shape[1]
    dist = np.zeros(num)
    for idx in range(num):
        corr = np.sum(u * v[:, idx].reshape(-1))
        norm1 = np.sum(u * u)
        norm2 = np.sum(v[:, idx].reshape(-1) * v[:, idx].reshape(-1))
        dist[idx] = (corr / np.sqrt(norm1 * norm2) + 1) / 2.0

    return dist


def isempty(np_array):
    """
    ****************************************************
    Directly taken from Nidhin's code
    ****************************************************

    Check if the passed in numpy array is empty

    Arguments:
        np_array {numppy.ndarray} -- numpy array.

    Returns:
        [bool] -- True if np_array is empty.
    """
    # change the position of if statement
    if np_array is None:
        return True

    np_array = np.asarray(np_array)
    if isinstance(np_array, dict):
        for key in np_array:
            np_array = np_array[key]
            break

    if np_array.size == 0:
        return True

    return False


def sub(s, I):
    """
        Returns a subset of the structure s
        s: dict with each element as numpy array
        I: numpy array containing index.
    """
    if isinstance(I, int) or I.shape == ():
        I = np.array([I])
    subset = {}
    for key in s:
        subset[key] = s[key][I, :]
    return subset


def apply_motion_prediction(fr_current, tracker):
    """
        apply motion models to predict the next locations of the targets
    """

    # apply motion model and predict next location
    dres = tracker.dres
    # change made [0] not used in last
    index = np.where(dres['state'] == 2)
    dres = sub(dres, index)
    cx = dres['x'] + dres['w'] / 2
    cy = dres['y'] + dres['h'] / 2
    fr = dres['fr'].astype(np.dtype('d'))

    # only use the past 10 frames
    num = len(fr)
    K = 10
    if num > K:
        cx = cx[num - K:num]
        cy = cy[num - K:num]
        fr = fr[num - K:num]

    fr_current = float(fr_current)

    # compute veloxity
    vx = 0
    vy = 0
    num = len(cx)
    count = 0
    for j in range(1, num):
        vx = vx + (cx[j] - cx[j - 1]) / (fr[j] - fr[j - 1])
        vy = vy + (cy[j] - cy[j - 1]) / (fr[j] - fr[j - 1])
        count += 1
    if count != 0:
        vx = vx / count
        vy = vy / count

    if len(cx) == 0:
        dres = tracker.dres
        cx_new = dres['x'][-1] + dres['w'][-1] / 2
        cy_new = dres['y'][-1] + dres['h'][-1] / 2
    else:
        cx_new = cx[-1] + vx * (fr_current + 1 - fr[-1])
        cy_new = cy[-1] + vy * (fr_current + 1 - fr[-1])

    prediction = [cx_new, cy_new]
    return np.array(prediction)


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
    """
    Same copied from nidhin's code
    lk optical flow implementation based on opencv

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


def tldPatch2Pattern(patch, patchsize):
    patch = imresize(patch, patchsize, 'bicubic')
    pattern = patch.transpose(1, 0).reshape(-1)
    pattern = pattern - np.mean(pattern)
    return pattern


def generate_pattern(img, bb, patchsize):
    """
        get patch under bounding box (bb), normalize it size, reshape to a column
        vector and normalize to zero mean and unit variance (ZMUV)
    """
    nBB = bb.shape[1]
    pattern = np.zeros((np.prod(patchsize), nBB))

    # for every bounding box
    for i in range(nBB):
        # sample patch
        patch = img_patch(img, bb[:, i])

        # normalize size to 'patchsize' and nomalize intensities to ZMUV
        pattern[:, i] = tldPatch2Pattern(patch, patchsize)
    return pattern


def imResample_data(I, imsize):
    '''
    ******************************************************
    Directly taken from nidhin's code
    *******************************************************
    :param I:
    :param imsize:
    :return:
    '''
    img = array.array('f', I.flatten('F'))
    ha = I.shape[0]
    wa = I.shape[1]
    hb = imsize[0]
    wb = imsize[1]

    I_scale = array.array('f', np.zeros(imsize).flatten('F'))
    imResample(img, I_scale, ha, hb, wa, wb, 1, 1.0)
    return np.around(np.array(I_scale)).astype(np.uint8).reshape(imsize[1], imsize[0]).T


def lk_crop_image_box(I, BB, tracker):
    s = [tracker.std_box[0] / bb_width(BB), tracker.std_box[1] / bb_height(BB)]

    bb_scale = np.around([BB[0] * s[0], BB[1] * s[1], BB[2] * s[0], BB[3] * s[1]])

    bb_scale[2] = bb_scale[0] + tracker.std_box[0] - 1

    bb_scale[3] = bb_scale[1] + tracker.std_box[1] - 1

    imsize = np.around([I.shape[0] * s[1], I.shape[1] * s[0]]).astype(np.int32)

    I_scale = imResample_data(I, imsize)
    bb_crop = bb_rescale_relative(bb_scale, tracker.enlarge_box)
    I_crop = im_crop(I_scale, bb_crop)
    BB_crop = bb_shift_absolute(bb_scale, [-bb_crop[0], -bb_crop[1]])
    return [I_crop, BB_crop, bb_crop, s]


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


def lk_initialize(tracker, frame_id, target_id, dres, ind, dres_image):
    x1 = dres['x'][ind]
    y1 = dres['y'][ind]
    x2 = dres['x'][ind] + dres['w'][ind]
    y2 = dres['y'][ind] + dres['h'][ind]

    # template num
    num = tracker.num
    tracker.target_id = target_id
    tracker.bb = np.zeros(shape=(4, 1), dtype=np.float64)

    # initialize all the templates
    bb = np.tile(np.reshape(np.array([x1, y1, x2, y2]), (4, 1)), (1, num))
    bb[:, 1] = bb_shift_relative(np.copy(bb[:, 0]), [-0.01, -0.01])
    bb[:, 2] = bb_shift_relative(np.copy(bb[:, 0]), [-0.01, 0.01])
    bb[:, 3] = bb_shift_relative(np.copy(bb[:, 0]), [0.01, -0.01])
    bb[:, 4] = bb_shift_relative(np.copy(bb[:, 0]), [0.01, 0.01])

    tracker.frame_ids = frame_id * np.ones(shape=(num, 1), dtype=np.int32)
    '''
    **********************************************
    check the shape below once transpose given in matlab code
    '''
    tracker.x1 = bb[0, :].reshape(-1, 1)
    tracker.y1 = bb[1, :].reshape(-1, 1)
    tracker.x2 = bb[2, :].reshape(-1, 1)
    tracker.y2 = bb[3, :].reshape(-1, 1)
    tracker.anchor = 0  # doubt

    # initialze the image for LK association
    tracker.Is = [None] * num
    tracker.BBs = [None] * num
    for i in range(num):
        I = dres_image['Igray'][tracker.frame_ids[i] - 1]  # doubt
        BB = [tracker.x1[i], tracker.y1[i], tracker.x2[i], tracker.y2[i]]

        # crop images and boxes
        I_crop, BB_crop, _, _ = lk_crop_image_box(I, BB, tracker)
        tracker.Is[i] = I_crop
        tracker.BBs[i] = BB_crop

    # Initialize the patterns
    img = dres_image['Igray'][frame_id - 1]
    tracker.patterns = generate_pattern(img, bb, tracker.patchsize)

    # box overlap history
    tracker.bb_overlaps = np.ones(shape=(num, 1))

    # tracker results
    tracker.bbs = [None] * num
    tracker.points = [None] * num
    tracker.flags = np.ones(shape=(num, 1))
    tracker.medFBs = np.zeros(shape=(num, 1))
    tracker.medFBs_left = np.zeros(shape=(num, 1))
    tracker.medFBs_right = np.zeros(shape=(num, 1))
    tracker.medFBs_up = np.zeros(shape=(num, 1))
    tracker.medFBs_down = np.zeros(shape=(num, 1))
    tracker.medNCCs = np.zeros(shape=(num, 1))
    tracker.overlaps = np.zeros(shape=(num, 1))
    tracker.scores = np.zeros(shape=(num, 1))
    tracker.indexes = np.zeros(shape=(num, 1), dtype=np.int32)
    tracker.nccs = np.zeros(shape=(num, 1))
    tracker.angles = np.zeros(shape=(num, 1))
    tracker.ratios = np.zeros(shape=(num, 1))

    if tracker.w_occluded is None:
        features = np.array([np.ones(shape=(1, tracker.fnum_occluded)), np.zeros(shape=(1, tracker.fnum_occluded))])
        labels = np.array([+1, -1])
        tracker.f_occluded = features
        tracker.l_occluded = labels
        tracker.w_occluded = svm_train(labels, features, '-c 1 -q -g 1 -b 1')

    return tracker


def lk_update(frame_id, tracker, img, dres_det, is_change_anchor):
    """update the LK tracker

    Arguments:
        frame_id {int} -- current frame id
        tracker {Tracker} -- tracker object
        img {} -- current image
        dres_det {[type]} -- images and bounding box
        is_change_anchor {bool} -- find the template with max FB error but not the anchor

    Returns:
        Tracker -- updated Tracker object
    """

    medFBs = tracker.medFBs
    if is_change_anchor == 0:
        # find the template with max FB error but not the anchor
        medFBs[tracker.anchor] = -np.inf
        index = np.argmax(medFBs)
    else:
        index = np.argmax(medFBs)
        tracker.anchor = index

    # update
    tracker.frame_ids[index] = frame_id
    tracker.x1[index] = tracker.bb[0]
    tracker.y1[index] = tracker.bb[1]
    tracker.x2[index] = tracker.bb[2]
    tracker.y2[index] = tracker.bb[3]
    tracker.patterns[:, index] = generate_pattern(img, tracker.bb, tracker.patchsize).reshape(-1)

    # update images and boxes
    BB = np.array([tracker.x1[index], tracker.y1[index], tracker.x2[index], tracker.y2[index]])
    I_crop, BB_crop, _, _ = lk_crop_image_box(img, BB, tracker)
    tracker.Is[index] = I_crop
    tracker.BBs[index] = BB_crop

    # compute overlap
    dres = {}
    dres['x'] = tracker.bb[0]
    dres['y'] = tracker.bb[1]
    dres['w'] = tracker.bb[2] - tracker.bb[0]
    dres['h'] = tracker.bb[3] - tracker.bb[1]
    num_det = len(dres_det['fr'])
    if not isempty(dres_det['fr']):
        o, _, _ = calc_overlap(dres, 0, dres_det, np.arange(0, num_det))
        tracker.bb_overlaps[index] = max(o)
    else:
        tracker.bb_overlaps[index] = 0

    return tracker


def lk_associate(frame_id, dres_image, dres_det, tracker):
    """use LK trackers for association

    Arguments:
        frame_id {int} -- current frame id.
        dres_image  -- image
        dres_det  -- detection images.
        tracker {Tracker} -- Tracker object.

    Returns:
        Tracker -- updated tracker.
    """

    # get cropped images and boxes
    J_crop = dres_det['I_crop'][0]
    BB2_crop = dres_det['BB_crop'][0]
    bb_crop_J = dres_det['bb_crop'][0]
    s_J = dres_det['scale'][0]

    for i in range(0, tracker.num):
        BB1 = np.array([tracker.x1[i], tracker.y1[i], tracker.x2[i], tracker.y2[i]])
        I_crop = tracker.Is[i]
        BB1_crop = tracker.BBs[i]

        # LK tracking
        BB3, xFJ, flag, medFB, medNCC, medFB_left, medFB_right, medFB_up, medFB_down = LK(I_crop, J_crop, BB1_crop,
                                                                                          BB2_crop, tracker.margin_box,
                                                                                          tracker.level)

        BB3 = bb_shift_absolute(BB3, np.array([bb_crop_J[0], bb_crop_J[1]]))
        BB3 = np.array([BB3[0] / s_J[0], BB3[1] / s_J[1], BB3[2] / s_J[0], BB3[3] / s_J[1]])

        ratio = (BB3[3] - BB3[1]) / (BB1[3] - BB1[1])
        ratio = min(ratio, 1 / ratio)

        if np.isnan(medFB) or np.isnan(medFB_left) or np.isnan(medFB_right) or np.isnan(medFB_up) or np.isnan(
                medFB_down) or np.isnan(medNCC) or ~bb_isdef(BB3):
            medFB = np.inf
            medFB_left = np.inf
            medFB_right = np.inf
            medFB_up = np.inf
            medFB_down = np.inf
            medNCC = 0
            o = 0
            score = 0
            ind = 0
            angle = 0
            flag = 2
            BB3 = np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            dres = {}
            # compute overlap
            dres['x'] = np.array([BB3[0]])
            dres['y'] = np.array([BB3[1]])
            dres['w'] = np.array([BB3[2] - BB3[0]])
            dres['h'] = np.array([BB3[3] - BB3[1]])
            o, _, _ = calc_overlap(dres, 0, dres_det, np.array([0]))
            ind = 0
            score = dres_det['r'][0]

            # compute angle
            centerI = np.array([(BB1[0] + BB1[2]) / 2, (BB1[1] + BB1[3]) / 2])
            centerJ = np.array([(BB3[0] + BB3[2]) / 2, (BB3[1] + BB3[3]) / 2])
            v = compute_velocity(tracker)
            v_new = np.array([centerJ[0] - centerI[0], centerJ[1] - centerI[1]]) / float(
                frame_id - tracker.frame_ids[i])
            if np.linalg.norm(v) > tracker.min_vnorm and np.linalg.norm(v_new) > tracker.min_vnorm:
                angle = np.dot(v, v_new) / (np.linalg.norm(v) * np.linalg.norm(v_new))
            else:
                angle = 1

        tracker.bbs[i] = BB3
        tracker.points[i] = xFJ
        tracker.flags[i] = flag
        tracker.medFBs[i] = medFB
        tracker.medFBs_left[i] = medFB_left
        tracker.medFBs_right[i] = medFB_right
        tracker.medFBs_up[i] = medFB_up
        tracker.medFBs_down[i] = medFB_down
        tracker.medNCCs[i] = medNCC
        tracker.overlaps[i] = o
        tracker.scores[i] = score
        tracker.indexes[i] = ind
        tracker.angles[i] = angle
        tracker.ratios[i] = ratio

    # combine tracking and detection results
    ind = np.argmin(tracker.medFBs)
    index = tracker.indexes[ind]
    # bb_det is column vector look in matlab code
    bb_det = np.array([dres_det['x'][index], dres_det['y'][index], dres_det['x'][index] + dres_det['w'][index],
                       dres_det['y'][index] + dres_det['h'][index]]).reshape(4, -1);

    if tracker.overlaps[ind] > tracker.overlap_box:
        # tracker.bb = np.mean(np.array(
        #     [np.repeat(tracker.bbs[ind].reshape(-1, 1), tracker.weight_tracking, axis=1), bb_det.reshape(-1, 1)]),
        #     axis=0)
        '''
        ********************************************************************************************
        check below line imp. one
        '''
        tracker.bb = np.mean(
            np.array([np.tile(tracker.bbs[ind].reshape(-1, 1), (1, tracker.weight_association)), bb_det]), axis=1)
        # for lstm
        # if tracker.is_lstm == 1:
        #   add_to_target_queue(tracker, frame_id, index, 0)
    else:
        tracker.bb = bb_det

    # compute pattern similarity
    if bb_isdef(tracker.bb):
        pattern = generate_pattern(dres_image['Igray'][frame_id - 1], tracker.bb, tracker.patchsize)
        nccs = distance(pattern, tracker.patterns)  # measure NCC to positive examples
        # changed to suit the dimensions
        tracker.nccs = nccs.reshape(-1, 1)
    else:
        tracker.nccs = np.zeros(shape=(tracker.num, 1))

    return tracker


def lk_tracking(frame_id, dres_image, dres_det, tracker):
    """Track the object based on lk tracking

    Arguments:
        frame_id  -- current frame id.
        dres_image  -- images
        dres_det  -- detection images
        tracker {} -- Tracker object

    Returns:
        [Tracker] -- updated tracker.
    """

    # current frame + motion
    J = dres_image['Igray'][frame_id - 1]
    ctrack = apply_motion_prediction(frame_id, tracker)
    w = tracker.dres['w'][-1]
    h = tracker.dres['h'][-1]
    BB3 = [ctrack[0] - w / 2, ctrack[1] - h / 2, ctrack[0] + w / 2, ctrack[1] + h / 2]
    [J_crop, BB3_crop, bb_crop, s] = lk_crop_image_box(J, BB3, tracker)

    num_det = len(dres_det['x'])
    for i in range(0, tracker.num):
        BB1 = [tracker.x1[i], tracker.y1[i], tracker.x2[i], tracker.y2[i]]
        I_crop = tracker.Is[i]
        BB1_crop = tracker.BBs[i]

        # LK tracking
        [BB2, xFJ, flag, medFB, medNCC, medFB_left, medFB_right, medFB_up, medFB_down] = LK(I_crop, J_crop, BB1_crop,
                                                                                            BB3_crop,
                                                                                            tracker.margin_box,
                                                                                            tracker.level_track)

        BB2 = bb_shift_absolute(BB2, [bb_crop[0], bb_crop[1]])
        BB2 = np.array([BB2[0] / s[0], BB2[1] / s[1], BB2[2] / s[0], BB2[3] / s[1]])

        ratio = (BB2[3] - BB2[1]) / (BB1[3] - BB1[1])
        ratio = min(ratio, 1 / ratio)

        if np.isnan(medFB) or np.isnan(medFB_left) or np.isnan(medFB_right) or np.isnan(medFB_up) or np.isnan(
                medFB_down) or np.isnan(medNCC) or ~bb_isdef(BB2) or ratio < tracker.max_ratio:
            medFB = np.inf
            medFB_left = np.inf
            medFB_right = np.inf
            medFB_up = np.inf
            medFB_down = np.inf
            medNCC = 0
            o = 0
            score = 0
            ind = 0
            angle = -1
            flag = 2
            BB2 = np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            # compute overlap
            dres = {}
            dres['x'] = np.array([BB2[0]])
            dres['y'] = np.array([BB2[1]])
            dres['w'] = np.array([BB2[2] - BB2[0]])
            dres['h'] = np.array([BB2[3] - BB2[1]])
            if len(dres_det['fr']) != 0:
                overlap, _, _ = calc_overlap(dres, 0, dres_det, np.arange(0, num_det))
                ind = np.argmax(overlap)
                o = overlap[ind]
                score = dres_det['r'][ind]
            else:
                o = 0
                score = -1
                ind = -1

            # compute angle
            centerI = np.array([(BB1[0] + BB1[2]) / 2, (BB1[1] + BB1[3]) / 2])
            centerJ = np.array([(BB2[0] + BB2[2]) / 2, (BB2[1] + BB2[3]) / 2])
            v = compute_velocity(tracker)
            v_new = np.array([centerJ[0] - centerI[0], centerJ[1] - centerI[1]]) / float(
                frame_id - tracker.frame_ids[i])
            if np.linalg.norm(v) > tracker.min_vnorm and np.linalg.norm(v_new) > tracker.min_vnorm:
                angle = np.dot(v, v_new) / (np.linalg.norm(v) * np.linalg.norm(v_new))
            else:
                angle = 1
        tracker.bbs[i] = BB2
        tracker.points[i] = xFJ
        tracker.flags[i] = flag
        tracker.medFBs[i] = medFB
        tracker.medFBs_left[i] = medFB_left
        tracker.medFBs_right[i] = medFB_right
        tracker.medFBs_up[i] = medFB_up
        tracker.medFBs_down[i] = medFB_down
        tracker.medNCCs[i] = medNCC
        tracker.overlaps[i] = o
        tracker.scores[i] = score
        tracker.indexes[i] = ind
        tracker.angles[i] = angle
        tracker.ratios[i] = ratio

    # combine tracking and detection results
    ind = tracker.anchor
    if tracker.overlaps[ind] > tracker.overlap_box:
        index = tracker.indexes[ind]
        bb_det = np.array([dres_det['x'][index], dres_det['y'][index], dres_det['x'][index] + dres_det['w'][index],
                           dres_det['y'][index] + dres_det['h'][index]]).reshape(-1, 1)
        # tracker.bb = np.mean(np.array(
        #     [np.repeat(tracker.bbs[ind].reshape(-1, 1), tracker.weight_tracking, axis=1), bb_det.reshape(-1, 1)]),
        #     axis=0)
        '''
        ********************************************************************************************
        check below line imp. one
        '''
        tracker.bb = np.mean(
            np.array([np.tile(tracker.bbs[ind].reshape(-1, 1), (1, tracker.weight_tracking)), bb_det]), axis=1)

        # # for lstm
        # if tracker.is_lstm == 1:
        #     add_to_target_queue(tracker, frame_id, dres_det['id'][index], 0)
    else:
        tracker.bb = tracker.bbs[ind].reshape(4, -1)

    # compute pattern similarity
    if bb_isdef(tracker.bb):
        pattern = generate_pattern(dres_image['Igray'][frame_id - 1], tracker.bb, tracker.patchsize)
        nccs = distance(pattern, tracker.patterns)  # measure NCC to positive examples
        tracker.nccs = nccs.reshape(-1, 1)
    else:
        tracker.nccs = np.zeros(shape=(tracker.num, 1))

    return tracker
