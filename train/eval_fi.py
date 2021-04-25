import cv2
import json
import math
import matplotlib.pyplot as plt
import os
import random
import torch

from dataset import depickle_data
from matplotlib.patches import Rectangle
from visualizer import *


## NOTE: only tested with 5 detections per frame, 5 frames per step
NUM_CANDIDATES_PER_FRAME = 5
NUM_FRAMES_PER_STEP = 5
NUM_CANDIDATES_PER_STEP = NUM_CANDIDATES_PER_FRAME * NUM_FRAMES_PER_STEP
DETECTION_EMBEDDING_DIM = 2048
BOUNDING_BOX_DIM = 4

NULL = '[unused1]'

FI = '/h/sagar/ece496-capstone/datasets/fi'
FI_VG = '/h/sagar/ece496-capstone/datasets/fi_datasets/YCII/VG/gnding_annot_all.json'

def read_json(path='output.json'):
    """
    Check for valid JSON format and read content
    path: path to JSON file
    """
    file = open(path)
    line = file.read().replace('\n', ' ')
    file.close()
    try:
        parsed_json = json.loads(line)
    except:
        assert False, 'Invalid JSON'
    return parsed_json


############### HELPER FUNCTIONS ###################
def compute_iou(bbox_a, bbox_b):
    """
    Return the IoU of bbox_a and bbox_b
    bbox_a: list of 4 positive integers representing x, y, width, height
    bbox_b: list of 4 positive integers representing x, y, width, height
    Returns iou: the IoU of bbox_a and bbox_b
    """

    a_x1 = bbox_a[0]            # x
    a_y1 = bbox_a[1]            # y
    a_x2 = a_x1 + bbox_a[2]     # x + width
    a_y2 = a_y1 + bbox_a[3]     # y + height

    b_x1 = bbox_b[0]            # x
    b_y1 = bbox_b[1]            # y
    b_x2 = b_x1 + bbox_b[2]     # x + width
    b_y2 = b_y1 + bbox_b[3]     # y + height

    #coordinates of intersected box
    int_x1 = max(a_x1, b_x1)
    int_y1 = max(a_y1, b_y1)
    int_x2 = min(a_x2, b_x2)
    int_y2 = min(a_y2, b_y2)

    if int_x1 > int_x2 or int_y1 > int_y2:
        return 0.0

    int_area = (int_y2 - int_y1)*(int_x2 - int_x1)
    a_area = (a_x2 - a_x1)*(a_y2 - a_y1)
    b_area = (b_x2 - b_x1)*(b_y2 - b_y1)

    union_area = a_area + b_area - int_area

    iou = int_area/float(union_area)

    return iou


def compute_iou_from_normalized_coords(bbox_normalized, img_width, img_height, bbox_b):
    """
    Return the IoU of bbox_normalized and bbox_b
    bbox_normalized: list of 4 positive integers representing normalized coordinates x1, y1, x2, y2
    img_width: width of image associated with bbox_normalized
    img_height: height of image associated with bbox_normalized
    bbox_b: list of 4 positive integers representing x, y, width, height
    Returns iou: the IoU of bbox_normalized and bbox_b
    """
    x1, y1 = bbox_normalized[0] * img_width, bbox_normalized[1] * img_height
    x2, y2 = bbox_normalized[2] * img_width, bbox_normalized[3] * img_height

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    bbox = [x1, y1, bbox_width, bbox_height]

    return compute_iou(bbox, bbox_b)


#given a list of vg keys, and a frame name the model grounded to, find the closest frame
#in time to that
#returns the bbox in that frame
def nearest_in_time(vg_keys_gt_ent, model_frame, gt_vid_bbox):
    #check the model_frame type
    if not isinstance(model_frame, int):
        if '/' in model_frame and '.jpg' in model_frame:
            frame_num = int(model_frame.split('/')[-1][:-4])
        elif '.jpg' in model_frame:
            frame_num = int(model_frame[:-4])
        else:
            frame_num = int(model_frame)
    else:
        #already an int
        frame_num = model_frame 

    min_diff = 999999
    best_bbox = None
    #best_bbox_formatted = None
    #best_frame = None
    
    for k in vg_keys_gt_ent:
        bbox_list = gt_vid_bbox[k]['bboxes']

        for bbox in bbox_list:
            gt_frame_num = int(bbox['img'].split('/')[-1][:-4])

            diff = abs(gt_frame_num - frame_num)

            if diff == 0:
                return bbox

            if diff < min_diff:
                min_diff = diff
                best_bbox = bbox
                #best_frame = bbox['img']
                
    #reorder to [x, y, w, h]
    #best_bbox_formatted = [best_bbox['bbox']['x'],
    #                       best_bbox['bbox']['y'],
    #                       best_bbox['bbox']['w'],
    #                       best_bbox['bbox']['h']]
    #print('here')
    return best_bbox #best_bbox_formatted, best_frame


#given the action id, and the ground truth entity number,
#returns the visual grounding key associated
#we return a list of strings corresponding to all instances for that entity
def get_vg_key(action_id, match_ent_id, gt_vid_bbox):
    #str_action_id = '('+action_id+', '+match_ent_id + ', ' + 0 + ')'

    #find all instances of this entity
    keys = [k for k in gt_vid_bbox.keys()]

    return [k for k in keys if eval(k)[0] == action_id and eval(k)[1] == match_ent_id]

############# EVALUATE, WITH VISUALIZATION############################

# TODO: this function is largely a duplication of the previous version of compute_eval_ious function. Need to refactor code.
#evaluates Mean IoU of model, and draws gt bboxes + model bboxes
#example run: VG, RR, mean_iou_pretrained = eval_im(model, num_actions=10, index=2, root=FI, gt_bbox_all=None)
def vis_eval_im(model, num_actions, index, root, gt_bbox_all):
    
    root = os.path.join(root, str(num_actions), str(index).zfill(5))
    pickles_root = os.path.join(root, 'pickles')
    frames_root = os.path.join(root, 'frames')

    frame_paths = depickle_data(pickles_root, 'frame_paths')
    entities = depickle_data(pickles_root, 'entities')
    actions = depickle_data(pickles_root, 'actions_list')
    actions.append('[NULL]')
    candidates = depickle_data(pickles_root, 'candidates')
    vid_id = depickle_data(pickles_root, 'vid_id')
    steps = depickle_data(pickles_root, 'steps')
    entity_count = depickle_data(pickles_root, 'entity_count')
    bboxes = torch.stack(list(zip(*candidates))[0]).squeeze(1).reshape(-1, BOUNDING_BOX_DIM)
    features = torch.stack(list(zip(*candidates))[1]).squeeze(1).reshape(-1, DETECTION_EMBEDDING_DIM)
    
    if gt_bbox_all is None:
        gt_bbox_all = read_json(FI_VG)
    
    gt_vid_bbox = gt_bbox_all[vid_id]
    
    VG, RR = model_inference(model, num_actions, steps, entities, entity_count, bboxes, features)
    
    # calculate mean grounding IoU
    #1) if gt doesn't have bbox, we skip (doesn't count towards IoU) - since model must ground all entities
    #2) we look for gt bbox in nearest frame to model grounded one
    # this may lead to addition leniency in IoU score
    mean_iou = 0.0
    num_ents = 0
    
    for action_idx, action_entities in enumerate(entities[:-1]):    
        print('--------------------------------------------------')
        print('Action {}: {}'.format(action_idx + 1, actions[action_idx]))

        if len(action_entities) == 0:
            print('No entities detected for this action.')

        frame_path = None
        prev_frame_path = None

        fig = None
        axes = None

        for ent_idx, entity in enumerate(action_entities):
            # VG processing.
            offset = NUM_FRAMES_PER_STEP * action_idx
            candidate = int(VG[action_idx][ent_idx])
            vg_idx = offset + math.floor(candidate / NUM_CANDIDATES_PER_FRAME)

            prev_frame_path = frame_path
            frame_path = frame_paths[vg_idx]
            bbox = bboxes[candidate]
            
            ################################################
            ## processing for ground truth entity bbox
            #index into gt as (action_idx, entity_idx, instance of entity)
            #str_action_id =  '('+str(action_idx)+', '+str(ent_idx) + ', ' + str(0) + ')'
            vg_keys = get_vg_key(action_idx, ent_idx, gt_vid_bbox)
            
            print(vg_keys)
            
            #gt_vid_box[str_action_id]['bboxes']
            gt_bbox = nearest_in_time(vg_keys_gt_ent=vg_keys, 
                                     model_frame=frame_path, 
                                     gt_vid_bbox=gt_vid_bbox)
            
            if gt_bbox is None:
                print('This entity has no ground truth bounding box')
                continue
            
            #gt_bbox_list, gt_frame
            gt_bbox_list = [gt_bbox['bbox']['x'],
                           gt_bbox['bbox']['y'],
                           gt_bbox['bbox']['w'],
                           gt_bbox['bbox']['h']]
            gt_frame = gt_bbox['img']
            #print(gt_frame)
            gt_box = Rectangle((gt_bbox_list[0], gt_bbox_list[1]), gt_bbox_list[2], gt_bbox_list[3], linewidth=1, edgecolor='red', facecolor='none')
            
            gt_bbox_width = gt_bbox_list[2]
            gt_bbox_height = gt_bbox_list[3]
            gt_x0 = gt_bbox_list[0]
            gt_y0 = gt_bbox_list[1]
            
            #print('GT frame is {}, model frame is {}'.format(gt_frame, frame_path))
            
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            if prev_frame_path != frame_path:
                fig = plt.figure()
                plt.imshow(frame)
                axes = plt.gca()

            x0, y0 = bbox[0] * frame_width, bbox[1] * frame_height
            x1, y1 = bbox[2] * frame_width, bbox[3] * frame_height

            bbox_width = x1 - x0
            bbox_height = y1 - y0
            
            vg_box = [x0, y0, bbox_width, bbox_height]
            
            box = Rectangle((x0, y0), bbox_width, bbox_height, linewidth=1, edgecolor='lime', facecolor='none')
            
            iou = compute_iou(vg_box, gt_bbox_list)
            print('IoU is: {}'.format(iou))
            
            mean_iou += iou
            num_ents +=1
            
            axes.add_patch(box)
            axes.add_patch(gt_box)
            axes.annotate(entity, (gt_x0 + (gt_bbox_width / 2), gt_y0 + (gt_bbox_height / 2)), color='white', fontsize=18, ha='center', va='center')

            # RR processing.
            rr_idx = int(RR[action_idx][ent_idx])
            print('\u001b[38;5;82m {} \u001b[38;5;208m -> Action {} ({}) \u001b[0m'.format(entity, rr_idx + 1, actions[rr_idx]))
        
        plt.show()
    
    mean_iou = mean_iou/num_ents
    return VG, RR, mean_iou


def compute_eval_ious(model, num_actions, index, root, gt_bbox_all, acc_thresh=0.5):
    """
    Return the mean IoUs and accuracy for model output, random, and best candidates
    model: model for inferencing
    num_actions: number of actions in video
    index: video index in dataset directory
    root: directory path to dataset base
    acc_thresh: IoU threshold for accuracy
    Returns mean_best_iou: upper bound mean IoU of candidate bboxes and ground truth bboxes
    Returns mean_rand_iou: mean IoU of randomly selected bboxes and ground truth bboxes
    Returns mean_ours_iou: mean IoU of model output bboxes and ground truth bboxes
    Returns best_acc: Top-1 accuracy for proposal upper bound
    Returns rand_acc: Top-1 accuracy for random selection
    Returns ours_acc: Top-1 accuracy for model output
    """
    
    # Load data from disk
    root = os.path.join(root, str(num_actions), str(index).zfill(5))
    pickles_root = os.path.join(root, 'pickles')
    frames_root = os.path.join(root, 'frames')
    frame_paths = depickle_data(pickles_root, 'frame_paths')
    entities = depickle_data(pickles_root, 'entities')
    actions = depickle_data(pickles_root, 'actions_list')
    actions.append('[NULL]')
    candidates = depickle_data(pickles_root, 'candidates')
    vid_id = depickle_data(pickles_root, 'vid_id')
    steps = depickle_data(pickles_root, 'steps')
    entity_count = depickle_data(pickles_root, 'entity_count')
    bboxes = torch.stack(list(zip(*candidates))[0]).squeeze(1).reshape(-1, BOUNDING_BOX_DIM)
    features = torch.stack(list(zip(*candidates))[1]).squeeze(1).reshape(-1, DETECTION_EMBEDDING_DIM)
    
    # Extract ground truth bbox info for entire video
    if gt_bbox_all is None:
        gt_bbox_all = read_json(FI_VG)
    gt_vid_bbox = gt_bbox_all[vid_id]
    
    # Run model inference
    VG, RR = model_inference(model, num_actions, steps, entities, entity_count, bboxes, features)
    
    # Calculate mean grounding IoU
    mean_best_iou = 0.0
    mean_rand_iou = 0.0
    mean_ours_iou = 0.0
    
    best_correct = 0 # number of correctly matched bboxes (based on given threshold)
    rand_correct = 0
    ours_correct = 0
    
    num_ents = 0
    
    for action_idx, action_entities in enumerate(entities[:-1]):        
        print('--------------------------------------------------')
        print('Action {}: {}'.format(action_idx + 1, actions[action_idx]))

        # if gt doesn't have bbox, skip (doesn't count towards IoU) - since model must ground all entities
        if len(action_entities) == 0:
            print('No entities detected for this action.')

        frame_path = None
        prev_frame_path = None

        # all candidates for the current action
        frame_candidate_bboxes = bboxes[NUM_CANDIDATES_PER_STEP*action_idx : (NUM_CANDIDATES_PER_STEP*(action_idx+1))]
        
        for ent_idx, entity in enumerate(action_entities):
            # VG processing
            offset = NUM_FRAMES_PER_STEP * action_idx
            candidate = int(VG[action_idx][ent_idx])
            vg_idx = offset + math.floor(candidate / NUM_CANDIDATES_PER_FRAME)

            prev_frame_path = frame_path
            frame_path = frame_paths[vg_idx]
            bbox = bboxes[candidate]
            
            ################################################
            ## processing for ground truth entity bbox
            #index into gt as (action_idx, entity_idx, instance of entity)
            #str_action_id =  '('+str(action_idx)+', '+str(ent_idx) + ', ' + str(0) + ')'
            vg_keys = get_vg_key(action_idx, ent_idx, gt_vid_bbox)
            
            # look for gt bbox in nearest frame to model output bbox - this may lead to addition leniency in IoU score
            #gt_vid_box[str_action_id]['bboxes']
            gt_bbox_info = nearest_in_time(vg_keys_gt_ent=vg_keys, 
                                      model_frame=frame_path, 
                                      gt_vid_bbox=gt_vid_bbox)
            
            if gt_bbox_info is None:
                print('This entity has no ground truth bounding box')
                continue
            
            #gt_frame, gt_bbox
            gt_frame = gt_bbox_info['img']
            gt_bbox = [gt_bbox_info['bbox']['x'], 
                       gt_bbox_info['bbox']['y'], 
                       gt_bbox_info['bbox']['w'], 
                       gt_bbox_info['bbox']['h']]
            #print(gt_frame)
            gt_box = Rectangle((gt_bbox[0], gt_bbox[1]), 
                               gt_bbox[2], 
                               gt_bbox[3], 
                               linewidth=1, 
                               edgecolor='red', 
                               facecolor='none')

            gt_x0 = gt_bbox[0]
            gt_y0 = gt_bbox[1]
            gt_bbox_width = gt_bbox[2]
            gt_bbox_height = gt_bbox[3]
            
            #print('GT frame is {}, model frame is {}'.format(gt_frame, frame_path))
            
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            # Calculate model output IoU
            ours_iou = compute_iou_from_normalized_coords(bbox, frame_width, frame_height, gt_bbox)
            print('Chosen Frame IoU: {}'.format(ours_iou))
            if ours_iou >= acc_thresh:
                ours_correct += 1

            # Calculate best IoU possible from all candidates for current action
            best_iou = 0.0
            for candidate_bbox in frame_candidate_bboxes:
                candidate_iou = compute_iou_from_normalized_coords(candidate_bbox, frame_width, frame_height, gt_bbox)
                best_iou = max(best_iou, candidate_iou)
            print('Best IoU possible = {}'.format(best_iou))
            if best_iou >= acc_thresh:
                best_correct += 1

            # Pick a random candidate from all candidates for current action
            rand_bbox = frame_candidate_bboxes[random.randint(0,NUM_CANDIDATES_PER_STEP-1)]
            rand_iou = compute_iou_from_normalized_coords(rand_bbox, frame_width, frame_height, gt_bbox)
            print('Random Candidate IoU = {}'.format(rand_iou))
            if rand_iou >= acc_thresh:
                rand_correct += 1

            mean_rand_iou += rand_iou
            mean_best_iou += best_iou
            mean_ours_iou += ours_iou

            num_ents += 1

    mean_best_iou /= num_ents
    mean_ours_iou /= num_ents
    mean_rand_iou /= num_ents
    
    print('Mean Upper Bound IoU: {}, Mean Random IoU: {}, Mean Model IoU: {}'.format(mean_best_iou, 
                                                                                     mean_rand_iou, 
                                                                                     mean_ours_iou))
    
    best_acc = best_correct / num_ents
    rand_acc = rand_correct / num_ents
    ours_acc = ours_correct / num_ents
    print('Top-1 acc@{}:\nProposal Upper Bound: {}, Random: {}, Model: {}'.format(acc_thresh, best_acc, rand_acc, ours_acc))
    
    return mean_best_iou, mean_rand_iou, mean_ours_iou, best_acc, rand_acc, ours_acc


def eval_all_dataset(model, acc_thresh=0.5):
    """
    Print VG evaluation scores for FI dataset
    model: model for inferencing
    acc_thresh: IoU threshold for accuracy
    """

    mean_best_ious = []
    mean_rand_ious = []
    mean_ours_ious = []

    all_vid_mean_best_iou = 0.0
    all_vid_mean_rand_iou = 0.0
    all_vid_mean_ours_iou = 0.0 # ours: model-chosen output
    
    all_vid_best_acc = 0.0
    all_vid_rand_acc = 0.0
    all_vid_ours_acc = 0.0

    vid_count = 0

    num_actions_list = [int(num_actions) for num_actions in sorted(os.listdir(FI))]

    for num_actions in num_actions_list:
        indices = len(os.listdir(os.path.join(FI, str(num_actions))))

        for idx in range(indices):
            mean_best_iou_vid, mean_rand_iou_vid, mean_ours_iou_vid, best_acc, rand_acc, ours_acc = compute_eval_ious(model, num_actions=num_actions, index=idx, root=FI, gt_bbox_all=None, acc_thresh=acc_thresh)

            mean_best_ious.append(mean_best_iou_vid)
            mean_rand_ious.append(mean_rand_iou_vid)
            mean_ours_ious.append(mean_ours_iou_vid)

            all_vid_mean_best_iou += mean_best_iou_vid
            all_vid_mean_rand_iou += mean_rand_iou_vid
            all_vid_mean_ours_iou += mean_ours_iou_vid
            
            all_vid_best_acc += best_acc
            all_vid_rand_acc += rand_acc
            all_vid_ours_acc += ours_acc
            
            vid_count += 1

    all_vid_mean_best_iou /= vid_count
    all_vid_mean_rand_iou /= vid_count
    all_vid_mean_ours_iou /= vid_count

    all_vid_best_acc /= vid_count
    all_vid_rand_acc /= vid_count
    all_vid_ours_acc /= vid_count
    
    print('--------------------------------------------------')
    print('EVALUATION SUMMARY')
    print('Number of videos: {}'.format(vid_count))
    print('Mean IoU:')
    print('\tProposal Upper Bound: {}'.format(all_vid_mean_best_iou))
    print('\tRandom: {}'.format(all_vid_mean_rand_iou))
    print('\tModel: {}'.format(all_vid_mean_ours_iou))
    
    print('Top-1 accuracy@{}:'.format(acc_thresh))
    print('\tProposal Upper Bound: {:.1f}%'.format(all_vid_best_acc*100))
    print('\tRandom: {:.1f}%'.format(all_vid_rand_acc*100))
    print('\tModel: {:.1f}%'.format(all_vid_ours_acc*100))
    print('--------------------------------------------------')
    
    return

