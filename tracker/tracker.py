import numpy as np
import pandas as pd
from mdp_function.mdp_utilities import *
from lk.lk_utilities import *
from svmutil import *
import cv2


def read_dres_image(args, seq_set, seq_name, logger):
    '''
    :param args: parser
    :param seq_set: train or test
    :param seq_name: name of the folder of which dres image will be built

    seq_num(in matlab code) is omitted since it will obtain no. of dres_image itself and produce the dres pickle file

    :return: dres_image which is dictionary containing images
    '''

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


def dataframetonumpy(df):
    """
        Convert the input label files(det, gt) to numpy from pandas dataframe.
    """
    numpy_data = {}
    for col in df:
        numpy_data[col] = df[col].as_matrix()
        # print(numpy_data[col].dtype)
    return numpy_data





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


class Tracker():
    def __init__(self, I, dres_det, labels, args, logger):
        image_width = I.shape[1]
        image_height = I.shape[0]

        # normalization factor for features
        self.image_width = image_width
        self.image_height = image_height
        self.max_width = max(dres_det['w'])
        self.max_height = max(dres_det['h'])
        self.max_score = max(dres_det['r'])
        self.fb_factor = args.fb_factor

        # active
        self.fnum_active = 6
        factive = mdp_feature_active(self, dres_det)
        index = np.nonzero(labels)
        self.factive = factive[index]
        self.lactive = labels[index]
        self.w_active = svm_train(self.lactive.tolist(), self.factive.tolist(), '-c 1 -q')

        # initial state
        self.prev_state = 1
        self.state = 1

        # association model
        self.fnum_tracked = 2
        self.fnum_occluded = 12

        self.w_occluded = []
        self.f_occluded = []
        self.l_occluded = []
        self.streak_occluded = 0

        # tracker parameters
        self.num = args.num
        self.threshold_ratio = args.threshold_ratio
        self.threshold_dis = args.threshold_dis
        self.threshold_box = args.threshold_box
        self.std_box = args.std_box  # [width height]
        self.margin_box = args.margin_box
        self.enlarge_box = args.enlarge_box
        self.level_track = args.level_track
        self.level = args.level
        self.max_ratio = args.max_ratio
        self.min_vnorm = args.min_vnorm
        self.overlap_box = args.overlap_box
        self.patchsize = args.patchsize
        self.weight_tracking = args.weight_tracking
        self.weight_association = args.weight_association

        # To display result
        self.is_show = args.is_show
