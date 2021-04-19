from dataset import depickle_data
from matplotlib.patches import Rectangle

import torch
import os
import math
import cv2
import matplotlib.pyplot as plt

NUM_FRAMES_PER_STEP = 5
DETECTION_EMBEDDING_SIZE = 2048
BOUNDING_BOX_SIZE = 4

NULL = '[unused1]'

def inference(model, num_actions, index, root, NUM_CANDIDATES_PER_FRAME = 20):
    root = "{}/{}/{}".format(root, num_actions, str(index).zfill(5))

    pickle_root = "{}/pickles".format(root)
    frames_root = "{}/frames".format(root)

    frame_paths = depickle_data(pickle_root, 'frame_paths')
    entities = depickle_data(pickle_root, 'entities')
    actions = depickle_data(pickle_root, 'actions_list')
    actions.append("[NULL]")
    candidates = depickle_data(pickle_root, 'candidates')

    steps = depickle_data(pickle_root, 'steps')
    entity_count = depickle_data(pickle_root, 'entity_count')
    bboxes = torch.stack(list(zip(*candidates))[0]).squeeze(1).reshape(-1, BOUNDING_BOX_SIZE)
    features = torch.stack(list(zip(*candidates))[1]).squeeze(1).reshape(-1, DETECTION_EMBEDDING_SIZE)

    VG, RR = model_inference(model, num_actions, steps, entities, entity_count, bboxes, features)
    
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

            box = Rectangle((x0, y0), bbox_width, bbox_height, linewidth=1, edgecolor='lime', facecolor='none')
            axes.add_patch(box)
            axes.annotate(entity, (x0 + (bbox_width / 2), y0 + (bbox_height / 2)), color='white', fontsize=18, ha='center', va='center')

            # RR processing.
            rr_idx = int(RR[a_idx][e_idx])
            print("\u001b[38;5;82m {} \u001b[38;5;208m -> Action {} ({}) \u001b[0m".format(entity, rr_idx + 1, actions[rr_idx]))
        
        plt.show()

    return VG, RR

def model_inference(model, num_actions, steps, entities, entity_count, bboxes, features):
    model.eval()
    
    steps = [steps]
    entity_count = [entity_count]
    entities = [entities]
    
    bboxes = bboxes.unsqueeze(0)
    features = features.unsqueeze(0)

    with torch.no_grad():
        _, _, _, _, _, VG, RR, _, _, _ = model(1, num_actions + 1, steps, features, bboxes, entity_count, entities)
        return VG.squeeze(0), RR.squeeze(0)
