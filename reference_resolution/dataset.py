import webvtt
import os
import glob
import re

from ling_obj_classes import * 
from torch.utils.data import Dataset, DataLoader

# Dataset for the YouCookII reference resolution (from https://finding-it.github.io).
class FI_YCII(Dataset):
    
    def __init__(self, data_dir='./FI-YCII'):
        self.data_dir = data_dir
        self.files = [file for file in glob.glob('{}/*/*'.format(data_dir))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        
        annotations = []
        action_steps = []

        # Generate action steps.
        for caption in webvtt.read(file):
            annotation = re.search('annot: (.*)\n?', caption.text).group(1)
            action_id = re.search('ACTID: (.*)\n?', caption.text).group(1)
            predicate = re.search('PRED: (.*)\n?', caption.text).group(1)
            
            # Number of direct objects ∈ [0, inf).
            direct_objects = re.search('\[DOBJ, .*\] (.*)?', caption.text)
            if direct_objects:
                direct_objects = self.make_entities(','.join(direct_objects.groups()), 'DOBJ')

            # Number of propositional phreases ∈ [0, inf).
            prop_phrases = re.search('\[PP, .*\] (.*)?', caption.text)
            if prop_phrases:
                prop_phrases = self.make_entities(','.join(prop_phrases.groups()), 'PP')
                            
            annotations.append(annotation)
            action_steps.append(ActionStep(action_id, predicate, direct_objects, prop_phrases))
                
        # Return both raw annotation texts and groudn truth action steps.
        return annotations, action_steps
    
    # Generate entity objects from (entity text, action ID) pairs.
    def make_entities(self, raw_entities, entity_type):
        raw_entities = raw_entities.split(',')
        raw_entities = [raw_entity.strip() for raw_entity in raw_entities]
        
        entities = []

        for raw_entity in raw_entities:
            split = raw_entity.rsplit(' ', 1)
            action_id = re.search('\((.*\d*)\)', raw_entity)
            
            # Sometimes entity texts or action IDs are null.
            if action_id and len(split) == 2:
                entities.append(Entity(split[0], entity_type, action_id.group(1)))
            elif action_id:
                entities.append(Entity(None, entity_type, action_id.group(1)))
            else:
                entities.append(Entity(raw_entity, entity_type, None))
                
        return entities

# This allows us to return custom objects from dataloader (not just tensors).
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

