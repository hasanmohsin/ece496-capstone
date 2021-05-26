# https://github.com/google-research/bert/issues/635
# https://colab.research.google.com/drive/18TyuMfZYlgQ_nXo-tr8LCnzUaoX0KS-h?usp=sharing#scrollTo=W4cZIVrg82ua.

import itertools
import torch
import einops
import torch.nn.functional as F

from transformers import LxmertModel, LxmertTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from entity_utils import *

class Model(nn.Module):
    DETECTION_EMBEDDING_SIZE = 2048
    OUTPUT_EMBEDDING_SIZE = 768

    def __init__(self, device, NUM_FRAMES_PER_STEP=5, MAX_DETECTIONS=20):
        super(Model, self).__init__()

        self.device = device

        self.NUM_FRAMES_PER_STEP = NUM_FRAMES_PER_STEP
        self.MAX_DETECTIONS = MAX_DETECTIONS

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert.to(device)

    def forward(self, BATCH_SIZE, NUM_ACTIONS, steps, features, boxes, entities, entity_count):
        '''
            NUM_ACTIONS: number of actions in video (excluding NULL step)
        '''
        CANDIDATES = self.NUM_FRAMES_PER_STEP * self.MAX_DETECTIONS

        features = features.to(self.device)
        boxes = boxes.to(self.device)
        
        inputs = self.lxmert_tokenizer(
            steps,
            padding="longest",
            truncation=False,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        inputs.input_ids = inputs.input_ids.to(self.device)
        inputs.attention_mask = inputs.attention_mask.to(self.device)
        inputs.token_type_ids = inputs.token_type_ids.to(self.device)

        output = self.lxmert(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=True
        )
        
        # Get indices of tokens corresponding to entities.
        entity_idx = get_ent_inds(self, entities, steps)

        # Extract both vision (candidate) and language (entity) embeddings.
        language_embeddings = get_entity_embeddings(output['language_output'], entity_idx)
        vision_embeddings = output['vision_output']        
        
        # Batch together the entities with padding.
        split_sizes = torch.tensor(entity_count).flatten().tolist()
        entity_embeddings = language_embeddings.split(split_sizes)
        
        E = pad_sequence(entity_embeddings, batch_first=True)
        E = E.reshape(BATCH_SIZE, NUM_ACTIONS, E.shape[1], self.OUTPUT_EMBEDDING_SIZE)
        
        MAX_ENTITIES = E.shape[2]
        
        V = vision_embeddings.reshape(BATCH_SIZE, NUM_ACTIONS, CANDIDATES, self.OUTPUT_EMBEDDING_SIZE)
        
        # Visual grounding score is the dot product between candidates and entities.
        VG_scores = torch.einsum('bacs, baes -> baec', V, E)
        VG_scores_max, VG_scores_index = VG_scores.max(dim=-1)
        
        # Alignment score is the visual grounding score across all steps and entities. Note
        # this is different from above which computes scores between entities and candidates
        # from the same step. This score is used for the alignment loss.
        #
        # Shape: BATCH * ENTITY_ACTION_ID * ENTITY * CANDIDATE_ACTION_ID * CANDIDATE
        alignment_scores = torch.einsum('bqcs, bwes -> bweqc', V, E)
        
        loss_data = (alignment_scores, entity_count, BATCH_SIZE, NUM_ACTIONS, MAX_ENTITIES)
        
        return loss_data, VG_scores_index, None
