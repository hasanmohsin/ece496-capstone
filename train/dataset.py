from torch.utils.data import Dataset, DataLoader

import torch
import glob
import os
import pickle

DETECTION_EMBEDDING_SIZE = 2048
BOUNDING_BOX_SIZE = 4
NUM_FRAMES_PER_STEP = 5
MAX_DETECTIONS = 20

def depickle_data(pickles_dir, fname):
    pickle_path = os.path.join(pickles_dir, fname + '.pickle')
    pickle_in = open(pickle_path, 'rb')
    data = pickle.load(pickle_in)
    return data
    
class YouCookII(Dataset):
    def __init__(self, num_actions, root):
        self.root = root
        self.path = '{}/{}'.format(root, num_actions)
        self.size = len([directory for directory in os.listdir(self.path)])
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        pickles_root = '{}/{}/pickles'.format(self.path, str(idx).zfill(5))
        
        video_id = depickle_data(pickles_root, 'vid_id')
        candidates = depickle_data(pickles_root, 'candidates')
        actions = depickle_data(pickles_root, 'actions_list')
        
        bboxes = torch.stack(list(zip(*candidates))[0]).squeeze(1).reshape(-1, BOUNDING_BOX_SIZE)
        features = torch.stack(list(zip(*candidates))[1]).squeeze(1).reshape(-1, DETECTION_EMBEDDING_SIZE)
        
        steps = depickle_data(pickles_root, 'steps')
        entities = depickle_data(pickles_root, 'entities')
        entity_count = depickle_data(pickles_root, 'entity_count')
        
        indices = depickle_data(pickles_root, 'indices')
        max_step_length = depickle_data(pickles_root, 'max_step_length')
        
        return video_id, bboxes, features, actions, steps, entities, entity_count, indices, max_step_length
    
def collate_fn(datapoints):
    video_id_list = []
    
    bboxes_tensor = None
    features_tensor = None
    
    steps_list = []
    entities_list = []
    entity_count_list = []
    
    indices_list = []
    max_step_length_list = []
        
    for i, data in enumerate(datapoints):
        video_id, bboxes, features, actions, steps, entities, entity_count, indices, max_step_length = data
        
        if i == 0:
            bboxes_tensor = torch.ones((len(datapoints), len(actions) * NUM_FRAMES_PER_STEP * MAX_DETECTIONS, BOUNDING_BOX_SIZE))
            features_tensor = torch.ones((len(datapoints), len(actions) * NUM_FRAMES_PER_STEP * MAX_DETECTIONS, DETECTION_EMBEDDING_SIZE))
        
        bboxes_tensor[i] = bboxes
        features_tensor[i] = features
        
        video_id_list.append(video_id)        
        
        steps_list.append(steps)
        entities_list.append(entities)
        entity_count_list.append(entity_count)
        
        indices_list.append(indices)
        max_step_length_list.append(max_step_length)
        
    return video_id_list, bboxes_tensor, features_tensor, steps_list, entities_list, entity_count_list, indices_list, max_step_length_list
