# Dependencies.

import webvtt
import os
import glob
import re

from ling_obj_classes import * 
from torch.utils.data import Dataset, DataLoader

class COINDataset(Dataset):
    
    def __init__(self, data_dir='./FI-YCII'):
        self.data_dir = data_dir
        self.files = [file for file in glob.glob('{}/*/*'.format(data_dir))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        
        annotations = []
        action_steps = []

        for caption in webvtt.read(file):
            annotation = re.search('annot: (.*)\n?', caption.text).group(1)
            action_id = re.search('ACTID: (.*)\n?', caption.text).group(1)
            predicate = re.search('PRED: (.*)\n?', caption.text).group(1)
            
            direct_objects = re.search('\[DOBJ, .*\] (.*)?', caption.text)
            if direct_objects:
                direct_objects = self.make_entities(','.join(direct_objects.groups()))

            prop_phrases = re.search('\[PP, .*\] (.*)?', caption.text)
            if prop_phrases:
                prop_phrases = self.make_entities(','.join(prop_phrases.groups()))
                            
            annotations.append(annotation)
            action_steps.append(ActionStep(action_id, predicate, direct_objects, prop_phrases))
                
        return annotations, action_steps
    
    def make_entities(self, raw_entities):
        raw_entities = raw_entities.split(',')
        raw_entities = [raw_entity.strip() for raw_entity in raw_entities]
        
        entities = []
        
        for raw_entity in raw_entities:
            split = raw_entity.rsplit(' ', 1)
            action_id = re.search('\((.*\d*)\)', split[0])
            
            if action_id and len(split) == 2:
                entities.append(Entity(split[0], 'DOBJ', split[1]))
            elif action_id:
                entities.append(Entity(None, 'DOBJ', action_id.group(1)))
            else:
                entities.append(Entity(raw_entity, 'DOBJ', None))
                
        return entities

def collate_fn(batch):
    return batch

# Example usage.

#dataset = COINDataset()
#loader = DataLoader(dataset, collate_fn=collate_fn)
#
#for data in loader:
#    print('------------------------')
#    for step in data[0][1]:
#        print(step)
#        print('')
#    print(data[0][0])

