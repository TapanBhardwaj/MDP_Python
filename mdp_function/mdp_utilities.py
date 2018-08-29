from tracker.tracker import *


def mdp_crop_image_box(dres, I, tracker):
    """
        add cropped image and box to dres
    """
    num = len(dres['fr'])
    dres['I_crop'] = [None] * num
    dres['BB_crop'] = [None] * num
    dres['bb_crop'] = [None] * num
    dres['scale'] = [None] * num

    for i in range(num):
        # todo change made .reshape added
        BB = np.array([dres['x'][i], dres['y'][i], dres['x'][i] + dres['w'][i], dres['y'][i] + dres['h'][i]]).reshape(
            -1, 1)
        I_crop, BB_crop, bb_crop, s = lk_crop_image_box(I, BB, tracker)

        dres['I_crop'][i] = I_crop
        dres['BB_crop'][i] = BB_crop
        dres['bb_crop'][i] = bb_crop
        dres['scale'][i] = s

    # todo doubt in this whether to covert them in array or not
    # Make numpy array to avoid error in sub function.
    for key in ['I_crop', 'BB_crop', 'bb_crop', 'scale']:
        dres[key] = np.array(dres[key])

    return dres


def mdp_feature_active(tracker, dres):
    """

    :param tracker:
    :param dres:
    :return:
    """
    num = len(dres['fr'])
    f = np.zeros(shape=(num, tracker.fnum_active))
    f[:, 0] = dres['x'] / tracker.image_width
    f[:, 1] = dres['y'] / tracker.image_height
    f[:, 2] = dres['w'] / tracker.max_width
    f[:, 3] = dres['h'] / tracker.max_height
    f[:, 4] = dres['r'] / tracker.max_score
    f[:, 5] = 1
    return f


def mdp_feature_occluded(frame_id, dres_image, dres, tracker):
    """Get the mdp features when the state is occluded.

    Arguments:
        frame_id {int} -- current frame number.
        dres_image {[type]} -- images
        dres {[type]} -- [description]
        tracker {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    f = np.zeros(shape=(1, tracker.fnum_occluded))
    m = len(dres['fr'])
    feature = np.zeros(shape=(m, tracker.fnum_occluded))
    flag = np.zeros(shape=(m, 1))
    for i in range(m):
        dres_one = sub(dres, i)
        tracker = lk_associate(frame_id, dres_image, dres_one, tracker)

        # design features
        # todo changes done remove [0] from last in next line
        index = np.where(tracker.flags != 2)
        if not isempty(index):
            f[0] = np.mean(np.exp(-tracker.medFBs[index] / tracker.fb_factor))
            f[1] = np.mean(np.exp(-tracker.medFBs_left[index] / tracker.fb_factor))
            f[2] = np.mean(np.exp(-tracker.medFBs_right[index] / tracker.fb_factor))
            f[3] = np.mean(np.exp(-tracker.medFBs_up[index] / tracker.fb_factor))
            f[4] = np.mean(np.exp(-tracker.medFBs_down[index] / tracker.fb_factor))
            f[5] = np.mean(tracker.medNCCs[index])
            f[6] = np.mean(tracker.overlaps[index])
            f[7] = np.mean(tracker.nccs[index])
            f[8] = np.mean(tracker.ratios[index])
            f[9] = tracker.scores[0] / tracker.max_score
            f[10] = dres_one['ratios'][0]
            f[11] = np.exp(-dres_one['distances'][0])
        else:
            f = np.zeros(shape=(1, tracker.fnum_occluded))

        feature[i, :] = f

        if len(np.where(tracker.flags != 2)[0]) == 0:
            flag[i] = 0
        else:
            flag[i] = 1

    return feature, flag


def mdp_feature_tracked(frame_id, dres_image, dres_det, tracker):
    """

    :param frame_id:
    :param dres_image:
    :param dres_det:
    :param tracker:
    :return:
    """
    # lk_tracked
    tracker = lk_tracking(frame_id, dres_image, dres_det, tracker)

    # extract features
    f = np.zeros(shape=(1, tracker.fnum_tracked))

    anchor = tracker.anchor
    f[0] = tracker.flags[anchor]
    f[1] = np.mean(tracker.bb_overlaps)

    return tracker, f


def mdp_initialize(I, dres_det, labels, args, logger):
    """

    :param I:
    :param dres_det:
    :param labels:
    :param args:
    :param logger:
    :return:
    """
    tracker = Tracker(I, dres_det, labels, args, logger)
    return tracker


def mdp_initialize_test(tracker, image_width, image_height, dres_det, logger):
    """
    Initialization for testing

    :param tracker:
    :param image_width:
    :param image_height:
    :param dres_det:
    :param logger:
    :return:
    """

    # normalization factor for features
    tracker.image_width = image_width
    tracker.image_height = image_height
    tracker.max_width = max(dres_det['w'])
    tracker.max_height = max(dres_det['h'])
    tracker.max_score = max(dres_det['r'])

    tracker.streak_tracked = 0

    # if tracker.is_lstm:
    #     logger.info('Using LSTM model')

    return tracker
