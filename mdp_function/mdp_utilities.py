import pickle

import pandas as pd
import copy
import logging

from lk.lk_utilities import *
from tracker.tracker import *

logger = logging.getLogger("MDP")


def read_dres_image(args, seq_set, seq_name, logger):
    """
    :param args: parser
    :param seq_set: train or test
    :param seq_name: name of the folder of which dres image will be built

    seq_num(in matlab code) is omitted since it will obtain no. of dres_image itself and produce the dres pickle file

    :return: dres_image which is dictionary containing images
    """

    image_root_path = os.path.join(args.data_dir, seq_set, seq_name, 'img1')
    print(image_root_path)
    num_images = len(
        [name for name in os.listdir(image_root_path) if os.path.isfile(os.path.join(image_root_path, name))])
    logger.info('Reading {} file(s)'.format(num_images))
    dres_image = {}
    dres_image['x'] = np.zeros(shape=(num_images, 1))
    dres_image['y'] = np.zeros(shape=(num_images, 1))
    dres_image['w'] = np.zeros(shape=(num_images, 1))
    dres_image['h'] = np.zeros(shape=(num_images, 1))
    dres_image['I'] = []
    dres_image['Igray'] = []
    for idx in range(0, num_images):
        # for detrac
        # image_path = os.path.join(image_root_path, 'img{:05}.jpg'.format(idx))
        image_path = os.path.join(image_root_path, '{:06}.jpg'.format(idx + 1))
        logger.info(image_path)
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        Igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dres_image['x'][idx] = 1
        dres_image['y'][idx] = 1
        dres_image['w'][idx] = w
        dres_image['h'][idx] = h
        dres_image['I'].append(image)
        dres_image['Igray'].append(Igray)

        return dres_image


def save(args, seq_name, dres_image, logger):
    """
        Save the input image files

    Arguments:
        args {[type]} -- [description]
        seq_name {[type]} -- [description]
        dres_image {[type]} -- [description]
        logger {[type]} -- [description]
    """

    filename = os.path.join(args.output_dir, seq_name + '_dres_image.pkl')
    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(dres_image, f)
    logger.info('Save file done')


def dataframetonumpy(df):
    """
        Convert the input label files(det, gt) to numpy from pandas dataframe.
    """
    numpy_data = {}
    for col in df:
        numpy_data[col] = df[col].as_matrix()
        # print(numpy_data[col].dtype)
    return numpy_data


def add_to_target_queue(tracker, frame, identity, is_text):
    """
    To create the lists for LSTM model

    :param tracker:
    :param frame:
    :param identity:
    :param is_text:
    :return:
    """
    # print('append', frame)
    tracker.prev_frames.append(frame)
    tracker.prev_det_id.append(identity)

    if len(tracker.prev_frames) > 6:
        tracker.prev_frames.pop(0)
        tracker.prev_det_id.pop(0)
    return tracker


def predict(model, args, data, logger):
    """

    :param model:
    :param args:
    :param data:
    :param logger:
    :return:
    """
    # logger.info('Starting prediction...')
    # stats = {'timer': utils.Timer()}

    # Make prediction
    loss, acc, batch_size, true_label, pred_label = model.predict(data)
    logger.info({'acc': acc, 'pred_label': pred_label})

    return {'acc': acc, 'pred_label': pred_label, 'features': model.get_features()}


def mdp_feature_occluded_lstm(frame_id, dres_image, dres, tracker, args):
    """

    :param frame_id:
    :param dres_image:
    :param dres:
    :param tracker:
    :param args:
    :return:
    """
    m = len(dres['fr'])
    features = np.zeros((m, tracker.fnum_occluded))
    print('prev', tracker.prev_frames)
    print('id', tracker.prev_det_id)
    for i in range(m):
        dres_one = sub(dres, i)

        targets = ['t1', 't2', 't3', 't4', 't5', 't6']
        identites = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6']

        data = {}

        data['directory'] = tracker.seq_name

        # Copy prev_frames
        for idx, frame in enumerate(tracker.prev_frames):
            data[targets[idx]] = int(frame)
        # Copy prev identity
        for idx, identity in enumerate(tracker.prev_det_id):
            data[identites[idx]] = int(identity)

        data['det_frame'] = int(dres_one['fr'])
        data['det_id'] = int(dres_one['id'])
        data['label'] = 1

        predict_data = args.predict_dataset.get(data)
        result = predict(args.model, args, predict_data, logger)
        # print(type(result['features'].view(-1).cpu().numpy()))
        features[i, :] = result['features'].view(-1).cpu().numpy()

    # print(type(features[0]))
    # print(m)
    return features, np.ones(m)


def concatenate_dres(dres1, dres2):
    """

    :param dres1:
    :param dres2:
    :return:
    """
    # todo check here
    if isempty(dres2['fr']):
        dres_new = copy.deepcopy(dres1)
    else:
        dres_new = {}
        for key in dres1:
            dres_new[key] = np.concatenate((dres1[key], dres2[key]))
    return dres_new


def dict_value_to_np(data):
    """

    :param data:
    :return:
    """
    for key in data:
        if isinstance(data[key], list):
            data[key] = np.array(data[key])
        elif isinstance(data[key], np.ndarray):
            continue
        else:
            data[key] = np.array([data[key]])
    return data


def interpolate_dres(dres1, dres2):
    """
        add dres2 to dres1 and interpolate
    """

    if isempty(dres2['fr']):
        dres_new = copy.deepcopy(dres1)

    index = np.where(dres1['state'] == 2)[0]

    if not isempty(index):
        ind = index[-1]
        fr1 = dres1['fr'][ind]
        fr2 = dres2['fr'][0]

        if fr2 - fr1 <= 5 and fr2 - fr1 > 1:
            dres1 = sub(dres1, np.arange(0, ind + 1))

            # box1
            x1 = dres1['x'][-1]
            y1 = dres1['y'][-1]
            w1 = dres1['w'][-1]
            h1 = dres1['h'][-1]
            r1 = dres1['r'][-1]

            # box2
            x2 = dres2['x'][0]
            y2 = dres2['y'][0]
            w2 = dres2['w'][0]
            h2 = dres2['h'][0]
            r2 = dres2['r'][0]

            # linear interpolation
            for fr in range(fr1 + 1, fr2):
                dres_one = sub(dres2, 0)
                dres_one['fr'] = fr
                dres_one['x'] = x1 + ((x2 - x1) / (fr2 - fr1)) * (fr - fr1)
                dres_one['y'] = y1 + ((y2 - y1) / (fr2 - fr1)) * (fr - fr1)
                dres_one['w'] = w1 + ((w2 - w1) / (fr2 - fr1)) * (fr - fr1)
                dres_one['h'] = h1 + ((h2 - h1) / (fr2 - fr1)) * (fr - fr1)
                dres_one['r'] = r1 + ((r2 - r1) / (fr2 - fr1)) * (fr - fr1)
                dres_one = dict_value_to_np(dres_one)

                for key in dres1:
                    dres1[key] = np.concatenate((dres1[key], dres_one[key]))
    # concatenate
    dres_new = {}
    for key in dres1:
        dres_new[key] = np.concatenate((dres1[key], dres2[key]))

    return dres_new


def mdp_value(tracker, frame_id, dres_image, dres_det, index_det, args):
    """

    :param tracker:
    :param frame_id:
    :param dres_image:
    :param dres_det:
    :param index_det:
    :param args:
    :return:
    """
    # tracked, decide to tracked or occluded
    if tracker.state == 2:
        # extract features with LK tracking
        tracker, f = mdp_feature_tracked(frame_id, dres_image, dres_det, tracker)

        dres_one = {}
        # build the dres structure
        if bb_isdef(tracker.bb):
            dres_one['fr'] = np.array([frame_id])
            dres_one['id'] = np.array([tracker.target_id])
            dres_one['x'] = tracker.bb[0]
            dres_one['y'] = tracker.bb[1]
            dres_one['w'] = tracker.bb[2] - tracker.bb[0]
            dres_one['h'] = tracker.bb[3] - tracker.bb[1]
            dres_one['r'] = np.array([1])
        else:
            dres_one = sub(tracker.dres, len(tracker.dres['fr']) - 1)
            dres_one['fr'] = np.array([frame_id])
            dres_one['id'] = np.array([tracker.target_id])

        # compute qscore
        qscore = 0
        if f[0] == 1 and f[1] > tracker.threshold_box:
            label = 1
        else:
            label = -1

        # make decision
        if label > 0:
            tracker.state = 2
            dres_one['state'] = np.array([2])
            tracker.dres = concatenate_dres(tracker.dres, dres_one)
            # update LK tracker
            tracker = lk_update(frame_id, tracker, dres_image['Igray'][frame_id - 1], dres_det, 0)
        else:
            tracker.state = 3
            dres_one['state'] = np.array([3])
            tracker.dres = concatenate_dres(tracker.dres, dres_one)

        tracker.prev_state = 2

    # occluded, decide to tracked or occluded
    elif tracker.state == 3:

        # association
        if isempty(index_det):
            qscore = 0
            label = -1
            f = []
        else:
            # extract features with LK association
            dres = sub(dres_det, index_det)
            if tracker.is_lstm == 1:
                features, flag = mdp_feature_occluded_lstm(frame_id, dres_image, dres, tracker, args)
            else:
                features, flag = mdp_feature_occluded(frame_id, dres_image, dres, tracker)

            # positive and all zero for negative
            m = features.shape[0]
            # todo change shape to (m,1)
            labels = -1 * np.ones((m, 1))
            # TODO: get the probability for svm
            if tracker.is_sk_svm:
                labels = tracker.svc_occluded.predict(features)
                probs = tracker.svc_occluded.predict_proba(features)
            else:
                labels, _, probs = svm_predict(labels.tolist(), features.tolist(), tracker.w_occluded, '-b 1 -q')
                probs = np.array(probs)
                labels = np.array(labels)

            probs[flag == 0, 0] = 0
            probs[flag == 0, 1] = 1
            labels[flag == 0] = -1

            ind = np.argmax(probs[:, 0])
            qscore = probs[ind, 0]
            label = labels[ind]
            f = features[ind, :]

            dres_one = sub(dres_det, index_det[ind])
            tracker = lk_associate(frame_id, dres_image, dres_one, tracker)

        # make a decision
        tracker.prev_state = tracker.state
        if label > 0:
            # association
            tracker.state = 2
            # build the dres structure
            dres_one = {}
            dres_one['fr'] = frame_id
            dres_one['id'] = tracker.target_id
            dres_one['x'] = tracker.bb[0]
            dres_one['y'] = tracker.bb[1]
            dres_one['w'] = tracker.bb[2] - tracker.bb[0]
            dres_one['h'] = tracker.bb[3] - tracker.bb[1]
            dres_one['r'] = 1
            dres_one['state'] = 2
            dres_one = dict_value_to_np(dres_one)

            if tracker.dres['fr'][-1] == frame_id:
                dres = tracker.dres
                index = np.arange(0, len(dres['fr']) - 1)
                tracker.dres = sub(dres, index)
            tracker.dres = interpolate_dres(tracker.dres, dres_one)

            # update LK tracker
            tracker = lk_update(frame_id, tracker, dres_image['Igray'][frame_id - 1], dres_det, 1)
        else:
            # no association
            tracker.state = 3
            dres_one = sub(tracker.dres, len(tracker.dres['fr']) - 1)
            dres_one['fr'] = frame_id
            dres_one['id'] = tracker.target_id
            dres_one['state'] = 3

            if tracker.dres['fr'][-1] == frame_id:
                dres = tracker.dres
                index = np.arange(0, len(dres['fr']) - 1)
                tracker.dres = sub(dres, index)

            dres_one = dict_value_to_np(dres_one)
            tracker.dres = concatenate_dres(tracker.dres, dres_one)
    return tracker, qscore, f


def mdp_reward_occluded(fr, f, dres_image, dres_gt, dres, index_det, tracker, args, is_text, logger):
    """

    :param fr:
    :param f:
    :param dres_image:
    :param dres_gt:
    :param dres:
    :param index_det:
    :param tracker:
    :param args:
    :param is_text:
    :param logger:
    :return:
    """
    is_end = 0
    label = 0

    # check if any detection ovelap with gt
    index = np.where(dres_gt['fr'] == fr)[0]
    if isempty(index):
        # todo changed from  =0 to = np.array([0])
        overlap = np.array([0])
    else:
        if dres_gt['covered'][index] > args.overlap_occ:
            overlap = np.array([0])
        else:
            overlap, _, _ = calc_overlap(dres_gt, index, dres, index_det)
    if is_text:
        logger.info('max overlap in association {:.2f}'.format(max(overlap)))

    if max(overlap) > args.overlap_pos:
        if tracker.state == 2:
            # if association is correct
            ov, _, _ = calc_overlap(dres_gt, index, tracker.dres, len(tracker.dres['fr']) - 1)
            if ov > args.overlap_pos:
                reward = 1
            else:
                reward = -1
                label = -1
                is_end = 1
                if is_text:
                    logger.info('association to wrong target ({:.2f}, {:.2f})! Game over'.format(max(overlap), ov[0]))
        else:
            # target not associated
            if dres_gt['covered'][index] == 0:
                if isempty(np.where(tracker.flags != 2)[0]):
                    reward = 0  # no update
                else:
                    reward = -1  # no association
                    label = 1
                    # extract features
                    ind = np.argmax(overlap)
                    dres_one = sub(dres, index_det[ind])
                    if tracker.is_lstm == 1:
                        features, flag = mdp_feature_occluded_lstm(fr, dres_image, dres_one, tracker, args)
                    else:
                        features, flag = mdp_feature_occluded(fr, dres_image, dres_one, tracker)

                    if is_text:
                        logger.info('Missed association')
                    is_end = 1
            else:
                reward = 1
    else:
        if tracker.state == 3:
            reward = 1
        else:
            ov, _, _ = calc_overlap(dres_gt, index, tracker.dres, len(tracker.dres['fr']) - 1)
            if ov < args.overlap_neg or max(overlap) < args.overlap_neg:
                reward = -1
                label = -1
                is_end = 1
                if is_text:
                    logger.info('associated to wrong target! Game over')
            else:
                reward = 0
    if is_text:
        logger.info('reward {:.1f}'.format(reward))
    return reward, label, f, is_end


def generate_association_index(tracker, frame_id, dres_det):
    """Get the index of imabounding box from detection to associate with current track.

    Arguments:
        tracker {Tracker} -- current Tracker object
        frame_id {int} -- current frame id
        dres_det -- detection images

    Returns:
        [type] -- [description]
    """

    ctrack = apply_motion_prediction(frame_id, tracker)

    num_det = len(dres_det['fr'])
    # todo doubt in .T from last
    cdets = np.array([dres_det['x'] + dres_det['w'] / 2, dres_det['y'] + dres_det['h'] / 2]).T

    # compute distances and aspect ratios
    distances = np.zeros(shape=(num_det, 1))
    ratios = np.zeros(shape=(num_det, 1))
    for i in range(num_det):
        distances[i] = np.linalg.norm(cdets[i, :] - ctrack) / tracker.dres['w'][-1]

        ratio = tracker.dres['h'][-1] / dres_det['h'][i]
        ratios[i] = min(ratio, 1 / ratio)
    # todo remove [0] from last in next line
    index_det = np.where(np.logical_and(distances < tracker.threshold_dis, ratios > tracker.threshold_ratio))
    dres_det['ratios'] = ratios
    dres_det['distances'] = distances

    return dres_det, index_det, ctrack


def read_mot2dres(filename):
    """
    Reading det and gt file
    :param filename:
    :return:
    """
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
    """
    Generate the training data for mdp_train.
    :param seq_name: basically folder name
    :param dres_image:
    :param args:
    :param logger:
    :return:
    """

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
            # todo reomve [0] from last
            index = np.where(dres_det['fr'] == fr)
            if len(index) != 0:
                overlap, _, _ = calc_overlap(dres, j, dres_det, index)
                ind = np.argmax(overlap)
                o = overlap[ind]
                dres['overlap'][j] = o
                dres['r'][j] = dres_det['r'][index[ind]]

                # area inside image
                # todo change in last parameter in function call below
                _, overlap, _ = calc_overlap(dres_det, index[ind], dres_image, fr)
                dres['area_inside'][j] = overlap

        # start with bounding overlap > args.overlap_pos and non-occluded box
        # todo remove [0] from the end
        index = np.where(np.logical_and(dres['overlap'] > args.overlap_pos, np.logical_and(dres['covered'] == 0, dres[
            'area_inside'] > args.exit_threshold)))
        if len(index) != 0:
            index_start = index[0]
            # todo change in the last parameter
            dres_train.append(sub(dres, np.array(range(index_start, num + 1))))

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
