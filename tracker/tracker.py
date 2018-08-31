import numpy as np
from mdp_function.mdp_utilities import *
from svmutil import *


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
