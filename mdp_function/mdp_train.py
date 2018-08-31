
from lk.lk_utilities import *
from tld.tld_utilities import *
from common.common_utilities import *
from mdp_function.mdp_utilities import *
import pickle
from tracker.tracker import *


def mdp_train(args, seq_name, tracker, logger):
    """MDP Training code

    Arguments:
        args {Namespace} -- MDP configuration
        seq_name {string} -- data folder name
        tracker {Tracker class} -- Tracker class object
        logger {logger} -- logger handle

    Returns:
        Tracker class -- Trained tracker class.
    """

    seq_set = 'train'

    # Read data
    filename = os.path.join(args.output_dir, seq_name + '_dres_image.pkl')
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            dres_image = pickle.load(f)
        logger.info('load images from file {} done'.format(filename))
    else:
        dres_image = read_dres_image(args, seq_set, seq_name, logger)
        save(args, seq_name, dres_image, logger)

    # Number of frame
    seq_num = len(dres_image['x'])

    # generate training data
    I = dres_image['Igray'][0]
    dres_train, dres_det, labels = generate_training_data(seq_name, dres_image, args, logger)

    # Initialize Tracker
    if tracker is None:
        logger.info('Initialize tracker from scratch')
        tracker = mdp_initialize(I, dres_det, labels, args, logger)

    args.logger = logger

    # print(type(args))
    # For each training sequence
    t = -1
    iter = 0
    # max_iter = args.max_iter
    max_count = args.max_count
    count = 0
    num_train = len(dres_train)
    # todo change shape from (num_train) to (num_train, 1)
    counter = np.zeros((num_train, 1))
    is_good = np.zeros((num_train, 1))
    is_difficult = np.zeros((num_train, 1))

    while True:
        iter += 1
        tracker.seq_name = seq_name
        logger.info('iter {}'.format(iter))

        if iter > args.max_iter:
            logger.info('Max iteration exceeds')
            break

        if len(np.where(is_good == 0)[0]) == 0:

            # two pass training
            if count == args.max_pass:
                break
            else:
                count += 1
                logger.info('***pass {} finished'.format(count))
                # todo change shape from (num_train) to (num_train, 1)
                is_good = np.zeros((num_train, 1))
                is_good[is_difficult == 1] = 1
                counter = np.zeros((num_train, 1))
                t = -1
        # find a sequence to train
        while True:
            t = t + 1
            if t >= num_train:
                t = 0
            if is_good[t] == 0:
                break

        if args.is_text:
            logger.info('Tracking sequence {}'.format(t))

        dres_gt = dres_train[t]

        # first frame
        fr = dres_gt['fr'][0]
        identity = dres_gt['id'][0]

        # reset tracker : Transfer the MDP to Tracked.
        tracker.prev_state = 1
        tracker.state = 1
        tracker.target_id = identity

        # For LSTM tracker
        tracker.prev_frames = []
        tracker.prev_det_id = []

        # start tracking
        while fr <= seq_num:
            if args.is_text:
                logger.info('frame {}, state {}'.format(fr, tracker.state))

            # extract detection
            # todo doubt
            index = np.where(dres_det['fr'] == fr)[0]
            dres = sub(dres_det, index)
            num_det = len(dres['fr'])

            # inactive
            if tracker.state == 0:
                if reward == 1:
                    is_good[t] = 1
                    logger.info('sequence {} is good'.format(t))
                break

            # active
            elif tracker.state == 1:

                # compute overlap
                overlap, _, _ = calc_overlap(dres_gt, 0, dres, np.arange(num_det))
                ind = np.argmax(overlap)
                ov = overlap[ind]
                if args.is_text:
                    logger.info('Start first frame overlap {:.2}'.format(ov))

                # initialize the LK tracker : Initial the target template
                tracker = lk_initialize(tracker, fr, identity, dres, ind, dres_image)
                tracker.state = 2
                tracker.streak_occluded = 0

                # build the dres structure
                dres_one = sub(dres, ind)
                tracker.dres = dres_one
                tracker.dres['id'] = np.array([tracker.target_id])
                tracker.dres['state'] = np.array([tracker.state])

                if tracker.is_lstm == 1:
                    tracker = add_to_target_queue(tracker, dres_one['fr'], dres_one['id'], args.is_text)

            # tracker
            elif tracker.state == 2:
                tracker.streak_occluded = 0
                # todo verify mdp_value
                tracker, _, _ = mdp_value(tracker, fr, dres_image, dres, [], args)

            # occluded
            elif tracker.state == 3:
                tracker.streak_occluded = tracker.streak_occluded + 1

                # find a set of detections for association
                dres = mdp_crop_image_box(dres, dres_image['Igray'][fr - 1], tracker)
                dres, index_det, ctrack = generate_association_index(tracker, fr, dres)
                index_gt = np.where(dres_gt['fr'] == fr)[0]
                if dres_gt['covered'][index_gt] != 0:
                    index_det = []
                tracker, _, f = mdp_value(tracker, fr, dres_image, dres, index_det, args)

                if not isempty(index_det):
                    # compute reward
                    reward, label, f, is_end = mdp_reward_occluded(fr, f, dres_image, dres_gt, dres, index_det, tracker,
                                                                   args, args.is_text, logger)

                    # update weights if negative reward
                    if reward == -1:
                        tracker.f_occluded = np.vstack((tracker.f_occluded, f))
                        tracker.l_occluded = np.append(tracker.l_occluded, label)
                        # todo what is happening
                        if args.is_sk_svm:
                            tracker.svc_occluded = tracker.svc_occluded.fit(tracker.f_occluded, tracker.l_occluded)
                        else:
                            tracker.w_occluded = svm_train(tracker.l_occluded.tolist(), tracker.f_occluded.tolist(),
                                                           '-c 1 -q -g 1 -b 1')
                        logger.info('training examples in occluded state {}'.format(tracker.f_occluded.shape[0]))
                    if is_end:
                        tracker.state = 0

                # Transition to inacitve if lost for a long time
                if tracker.streak_occluded > args.max_occlusion:
                    tracker.state = 0
                    if len(np.where(dres_gt['fr'] == fr)[0]) == 0:
                        reward = 1
                    logger.info('Target exits due to long time occlusion')

            # check if outside image
            if tracker.state == 2:
                _, ov, _ = calc_overlap(tracker.dres, tracker.dres['fr'].shape[0] - 1, dres_image, fr - 1)
                if ov < args.exit_threshold:
                    logger.info('Target outside image by checking boarders')
                    tracker.state = 0
                    reward = 1

            # try to connect recently lost target
            if not (tracker.state == 3 and tracker.prev_state == 2):
                fr = fr + 1

        if fr > seq_num:
            is_good[t] = 1
            logger.info('sequence {} is good'.format(t))
        counter[t] = counter[t] + 1
        if counter[t] > max_count:
            is_good[t] = 1
            is_difficult[t] = 1
            logger.info('sequence {} mac iteration'.format(t))
    logger.info('Finish training {}'.format(seq_name))

    # save model
    if args.is_save:
        # save SVM
        filename = os.path.join(args.output_dir, args.name, seq_name + '_w_active')
        svm_save_model(filename, tracker.w_active)
        filename = os.path.join(args.output_dir, args.name, seq_name + '_w_occluded')
        svm_save_model(filename, tracker.w_occluded)
        w_active = tracker.w_active
        w_occluded = tracker.w_occluded
        tracker.w_active = None
        tracker.w_occluded = None

        filename = os.path.join(args.output_dir, args.name, seq_name + '_tracker.pkl')
        logger.info('Saving the tracker at {}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(tracker, f)
        tracker.w_active = w_active
        tracker.w_occluded = w_occluded

    return tracker


# if __name__ == '__main__':
#     # Parse cmdline args and setup environment
#     parser = argparse.ArgumentParser(
#         'Appearance Model',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     config.add_args(parser)
#     args = parser.parse_args()
#     mdp_train()
