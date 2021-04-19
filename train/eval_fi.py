import random
from dataset import depickle_data
from matplotlib.patches import Rectangle

import torch
import os
import math
import cv2
import matplotlib.pyplot as plt
import json

from visualizer import *

## NOTE: only tested with 5 detections per frame, 5 frames per step
NUM_FRAMES_PER_STEP = 5
NUM_CANDIDATES_PER_FRAME = 5
DETECTION_EMBEDDING_SIZE = 2048
BOUNDING_BOX_SIZE = 4

NULL = '[unused1]'

FI = "/h/sagar/ece496-capstone/datasets/fi"

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
    Computes the IoU of bbox_a and bbox_b
    bbox_a: list of 4 positive integers representing x, y, width, height
    bbox_b: list of 4 positive integers representing x, y, width, height
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

#evaluates Mean IoU of model, and draws gt bboxes + model bboxes
#example run: VG, RR, mean_iou_pretrained = eval_im(model, num_actions=10, index=2, root=FI, gt_bbox_all=None)
def vis_eval_im(model, num_actions, index, root, gt_bbox_all):
    
    root = "{}/{}/{}".format(root, num_actions, str(index).zfill(5))

    pickle_root = "{}/pickles".format(root)
    frames_root = "{}/frames".format(root)

    frame_paths = depickle_data(pickle_root, 'frame_paths')
    entities = depickle_data(pickle_root, 'entities')
    actions = depickle_data(pickle_root, 'actions_list')
    actions.append("[NULL]")
    candidates = depickle_data(pickle_root, 'candidates')
    vid_id = depickle_data(pickle_root, 'vid_id')
    
    steps = depickle_data(pickle_root, 'steps')
    entity_count = depickle_data(pickle_root, 'entity_count')
    bboxes = torch.stack(list(zip(*candidates))[0]).squeeze(1).reshape(-1, BOUNDING_BOX_SIZE)
    features = torch.stack(list(zip(*candidates))[1]).squeeze(1).reshape(-1, DETECTION_EMBEDDING_SIZE)
    
    if gt_bbox_all is None:
        gt_bbox_all = read_json("/h/sagar/ece496-capstone/datasets/fi_datasets/YCII/VG/gnding_annot_all.json")
    
    gt_vid_bbox = gt_bbox_all[vid_id]
    
    VG, RR = model_inference(model, num_actions, steps, entities, entity_count, bboxes, features)
    
    #calculate mean groundin IoU
    #1) if gt doesn't have bbox, we skip (doesn't count towards IoU) - since model must ground all entities
    #2) we look for gt bbox in nearest frame to model grounded one
    # this may lead to addition leniency in IoU score
    mean_iou = 0.0
    num_ents = 0
    
    for a_idx, action in enumerate(entities[:-1]):        
        print("Action {}: {}".format(a_idx + 1, actions[a_idx]))
        print("-------------------------------")

        if len(action) == 0:
            print("No entities detected for this action.")

        frame_path = None
        prev_frame_path = None

        fig = None
        axes = None

        for e_idx, entity in enumerate(action):
            # VG processing.
            offset = NUM_FRAMES_PER_STEP * a_idx
            candidate = int(VG[a_idx][e_idx])
            vg_idx = offset + math.floor(candidate / NUM_CANDIDATES_PER_FRAME)

            prev_frame_path = frame_path
            frame_path = frame_paths[vg_idx]
            bbox = bboxes[candidate]
            
            ################################################
            ## processing for ground truth entity bbox
            #index into gt as (action_idx, entity_idx, instance of entity)
            #str_action_id =  '('+str(a_idx)+', '+str(e_idx) + ', ' + str(0) + ')'
            vg_keys = get_vg_key(a_idx, e_idx, gt_vid_bbox)
            
            print(vg_keys)
            
            #gt_vid_box[str_action_id]['bboxes']
            gt_bbox = nearest_in_time(vg_keys_gt_ent=vg_keys, 
                                     model_frame=frame_path, 
                                     gt_vid_bbox=gt_vid_bbox)
            
            if gt_bbox is None:
                print("This entity has no ground truth bounding box")
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
            
            #print("GT frame is {}, model frame is {}".format(gt_frame, frame_path))
            
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
            print("IoU is: {}".format(iou))
            
            mean_iou += iou
            num_ents +=1
            
            axes.add_patch(box)
            axes.add_patch(gt_box)
            axes.annotate(entity, (gt_x0 + (gt_bbox_width / 2), gt_y0 + (gt_bbox_height / 2)), color='white', fontsize=18, ha='center', va='center')

            # RR processing.
            rr_idx = int(RR[a_idx][e_idx])
            print("\u001b[38;5;82m {} \u001b[38;5;208m -> Action {} ({}) \u001b[0m".format(entity, rr_idx + 1, actions[rr_idx]))
        
        plt.show()
    
    mean_iou = mean_iou/num_ents
    return VG, RR, mean_iou


## evaluates video, and computes:
# 1) mean model IoU
# 2) IoU achieved by random selection of a candidate in available frames of a step
# 3) best IoU possible with candidate bboxes in video
#compute iou over all candidates
def compute_eval_ious(model, num_actions, index, root, gt_bbox_all):
    
    root = "{}/{}/{}".format(root, num_actions, str(index).zfill(5))

    pickle_root = "{}/pickles".format(root)
    frames_root = "{}/frames".format(root)

    frame_paths = depickle_data(pickle_root, 'frame_paths')
    entities = depickle_data(pickle_root, 'entities')
    actions = depickle_data(pickle_root, 'actions_list')
    actions.append("[NULL]")
    candidates = depickle_data(pickle_root, 'candidates')
    vid_id = depickle_data(pickle_root, 'vid_id')
    
    steps = depickle_data(pickle_root, 'steps')
    entity_count = depickle_data(pickle_root, 'entity_count')
    bboxes = torch.stack(list(zip(*candidates))[0]).squeeze(1).reshape(-1, BOUNDING_BOX_SIZE)
    features = torch.stack(list(zip(*candidates))[1]).squeeze(1).reshape(-1, DETECTION_EMBEDDING_SIZE)
    
    if gt_bbox_all is None:
        gt_bbox_all = read_json("/h/sagar/ece496-capstone/datasets/fi_datasets/YCII/VG/gnding_annot_all.json")
    
    gt_vid_bbox = gt_bbox_all[vid_id]
    
    VG, RR = model_inference(model, num_actions, steps, entities, entity_count, bboxes, features)
    
    #calculate mean groundin IoU
    #1) if gt doesn't have bbox, we skip (doesn't count towards IoU) - since model must ground all entities
    #2) we look for gt bbox in nearest frame to model grounded one
    # this may lead to addition leniency in IoU score
    mean_chosen_iou = 0.0
    mean_best_iou = 0.0
    mean_rand_iou = 0.0
    
    num_ents = 0
    
    for a_idx, action in enumerate(entities[:-1]):        
        print("Action {}: {}".format(a_idx + 1, actions[a_idx]))
        print("-------------------------------")

        if len(action) == 0:
            print("No entities detected for this action.")

        frame_path = None
        prev_frame_path = None

        fig = None
        axes = None

        #the possible bboxes available for grounding at this frame
        frame_candidate_bboxes = bboxes[25*a_idx:(25*a_idx + 25)]
        
        for e_idx, entity in enumerate(action):
            # VG processing.
            offset = NUM_FRAMES_PER_STEP * a_idx
            candidate = int(VG[a_idx][e_idx])
            vg_idx = offset + math.floor(candidate / NUM_CANDIDATES_PER_FRAME)

            prev_frame_path = frame_path
            frame_path = frame_paths[vg_idx]
            bbox = bboxes[candidate]
            
            ################################################
            ## processing for ground truth entity bbox
            #index into gt as (action_idx, entity_idx, instance of entity)
            #str_action_id =  '('+str(a_idx)+', '+str(e_idx) + ', ' + str(0) + ')'
            vg_keys = get_vg_key(a_idx, e_idx, gt_vid_bbox)
            
            #gt_vid_box[str_action_id]['bboxes']
            gt_bbox = nearest_in_time(vg_keys_gt_ent=vg_keys, 
                                     model_frame=frame_path, 
                                     gt_vid_bbox=gt_vid_bbox)
            
            if gt_bbox is None:
                print("This entity has no ground truth bounding box")
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
            
            #print("GT frame is {}, model frame is {}".format(gt_frame, frame_path))
            
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            x0, y0 = bbox[0] * frame_width, bbox[1] * frame_height
            x1, y1 = bbox[2] * frame_width, bbox[3] * frame_height

            bbox_width = x1 - x0
            bbox_height = y1 - y0
            
            vg_box = [x0, y0, bbox_width, bbox_height]
            
       
            iou = compute_iou(vg_box, gt_bbox_list)
            print("Chosen Frame's IoU is: {}".format(iou))
            
            best_iou = 0.0
            
            #iterate over all possible choices to find best iou possible
            for c_bbox in frame_candidate_bboxes:
                c_x0, c_y0 = c_bbox[0] * frame_width, c_bbox[1] * frame_height
                c_x1, c_y1 = c_bbox[2] * frame_width, c_bbox[3] * frame_height

                c_bbox_width = c_x1 - c_x0
                c_bbox_height = c_y1 - c_y0

                c_box = [c_x0, c_y0, c_bbox_width, c_bbox_height]
                
                c_iou = compute_iou(c_box, gt_bbox_list)
                
                if c_iou > best_iou:
                    best_iou = c_iou
              
            print("Best IoU possible = {}".format(best_iou))
            
            #pick random candidate of the 5*5 available
            rand_ind = random.randint(0,24)
            c_bbox = frame_candidate_bboxes[rand_ind]
            #iterate over all possible choices to find best iou possible
              
            c_x0, c_y0 = c_bbox[0] * frame_width, c_bbox[1] * frame_height
            c_x1, c_y1 = c_bbox[2] * frame_width, c_bbox[3] * frame_height

            c_bbox_width = c_x1 - c_x0
            c_bbox_height = c_y1 - c_y0

            c_box = [c_x0, c_y0, c_bbox_width, c_bbox_height]

            c_iou = compute_iou(c_box, gt_bbox_list)

              
            print("Random Candidate IoU = {}".format(c_iou))
            
            mean_rand_iou +=c_iou
            mean_best_iou += best_iou
            mean_chosen_iou += iou
            num_ents +=1
            
    mean_best_iou /= num_ents
    mean_chosen_iou /= num_ents
    mean_rand_iou /= num_ents
    
    print("Mean Chosen IoU: {}, Random IoU: {}, Best IoU: {}".format(mean_chosen_iou, mean_rand_iou, mean_best_iou))
    
    return mean_chosen_iou, mean_rand_iou, mean_best_iou


def eval_all_dataset(model):
    #EVAL over all FI dataset
    num_action_list = [10, 11,12,13,15,17,18,19, 21, 23, 4, 8, 9]

    mean_chosen_ious = []
    mean_rand_ious = []
    mean_best_ious = []

    all_vid_mean_chosen_iou = 0.0
    all_vid_mean_rand_iou = 0.0
    all_vid_mean_best_iou = 0.0

    num_vid=  0

    for num_action in num_action_list:
        indices = len(os.listdir(FI+'/'+str(num_action)))
        #print(indices)
        for idx in range(indices):
            mean_chosen_iou_vid, mean_rand_iou_vid, mean_best_iou_vid =  compute_eval_ious(model, num_actions=num_action, index=idx, root=FI, gt_bbox_all=None)

            mean_chosen_ious.append(mean_chosen_iou_vid)
            mean_best_ious.append(mean_best_iou_vid)
            mean_rand_ious.append(mean_rand_iou_vid)

            all_vid_mean_chosen_iou += mean_chosen_iou_vid
            all_vid_mean_rand_iou += mean_rand_iou_vid
            all_vid_mean_best_iou += mean_best_iou_vid


            num_vid +=1

    all_vid_mean_chosen_iou /=num_vid

    all_vid_mean_rand_iou /=num_vid

    all_vid_mean_best_iou /=num_vid

  
    print("All vids - Mean Chosen IoU: {}, Random IoU: {}, Best IoU: {}, num Videos: {}".format(all_vid_mean_chosen_iou, 
                                                                                                all_vid_mean_rand_iou, 
                                                                                                all_vid_mean_best_iou,
                                                                                               num_vid))
                
    return