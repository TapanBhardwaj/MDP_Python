from tracker.tracker import *


def read_mot2dres(filename):
    '''
    Reading det and gt file
    :param filename:
    :return:
    '''
    data = pd.read_csv(filename, names=['fr', 'id', 'x', 'y', 'w', 'h', 'r', 'd1', 'd2', 'd3'])
    data.drop(columns=['d1', 'd2', 'd3'], inplace=True)
    data = dataframetonumpy(data)
    for key in ['x', 'y', 'w', 'h', 'r']:
        data[key] = data[key].astype(np.dtype('d'))
    return data

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


def generate_training_data(seq_name, dres_image, args, logger):
    '''
    Generate the training data for mdp_train.
    :param seq_name: basically folder name
    :param dres_image:
    :param args:
    :param logger:
    :return:
    '''


    seq_set = 'train'

    # read detections
    filename = os.path.join(args.data_dir, seq_set, seq_name, 'det', 'det.txt')
    dres_det = read_mot2dres(filename)

    # read ground truth
    filename = os.path.join(args.data_dir, seq_set, seq_name, 'gt', 'gt.txt')
    dres_gt = read_mot2dres(filename)
    y_gt = dres_gt['y'] + dres_gt['h']
    # print(y_gt)

    # collect true positives and false alarms from detections
    num = len(dres_det['fr'])
    labels = np.zeros(shape=(num, 1), dtype='int64')
    overlaps = np.zeros(shape=(num, 1))
    # print(labels)

    for i in range(num):
        fr = dres_det['fr'][i]
        # todo remove [0] from last
        index = np.where(dres_gt['fr'] == fr)
        # print(index)
        if index.shape[0] != 0:
            overlap, _, _ = calc_overlap(dres_det, i, dres_gt, index)
            o = max(overlap)
            if o < args.overlap_neg:
                labels[i] = -1
            elif o > args.overlap_pos:
                labels[i] = 1
            else:
                labels[i] = 0
            overlaps[i] = o
        else:
            overlaps[i] = 0
            # todo change form 0 to -1
            labels[i] = -1

    # build the training sequences
    ids = np.unique(dres_gt['id'])
    dres_train = []
    count = 0
    for i in range(ids.shape[0]):
        # todo remove [0] from last
        index = np.where(dres_gt['id'] == ids[i])
        dres = sub(dres_gt, index)

        # check if the target is occluded or not
        num = len(dres['fr'])
        dres['occluded'] = np.zeros(shape=(num, 1))
        dres['covered'] = np.zeros(shape=(num, 1))
        dres['overlap'] = np.zeros(shape=(num, 1))
        dres['r'] = np.zeros(shape=(num, 1))
        dres['area_inside'] = np.zeros(shape=(num, 1))
        y = dres['y'] + dres['h']

        for j in range(num):
            fr = dres['fr'][j]
            # todo remove [0] from last
            index = np.where(np.logical_and(dres_gt['fr'] == fr, dres_gt['id'] != ids[i]))

            if len(index) != 0:
                _, ov, _ = calc_overlap(dres, j, dres_gt, index)
                ov[y[j] > y_gt[index]] = 0
                dres['covered'][j] = max(ov)

            if dres['covered'][j] > args.overlap_occ:
                dres['occluded'][j] = 1

            # overlap with detections
            #todo reomve [0] from last
            index = np.where(dres_det['fr'] == fr)
            if len(index) != 0:
                overlap, _, _ = calc_overlap(dres, j, dres_det, index)
                ind = np.argmax(overlap)
                o = overlap[ind]
                dres['overlap'][j] = o
                dres['r'][j] = dres_det['r'][index[ind]]

                # area inside image
                # todo change in last parameter in function call below
                _, overlap, _ = calc_overlap(dres_det, index[ind], dres_image,fr)
                dres['area_inside'][j] = overlap

        # start with bounding overlap > args.overlap_pos and non-occluded box
        # todo remove [0] from the end
        index = np.where(np.logical_and(dres['overlap'] > args.overlap_pos, np.logical_and(dres['covered'] == 0, dres[
            'area_inside'] > args.exit_threshold)))
        if len(index) != 0:
            index_start = index[0]
            # todo change in the last parameter
            dres_train.append(sub(dres, np.array(range(index_start, num+1))))

    return (dres_train, dres_det, labels)


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
