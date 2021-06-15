# https://github.com/google-research/bert/issues/635
# https://colab.research.google.com/drive/18TyuMfZYlgQ_nXo-tr8LCnzUaoX0KS-h?usp=sharing#scrollTo=W4cZIVrg82ua.

import itertools
import torch
import einops
import torch.nn.functional as F

from transformers import LxmertModel, LxmertTokenizer
from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from entity_utils import *
from loss import *
from accuracy import *
from eval_fi import eval_all_dataset

# Lightning module imports.

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
    
class LitModel(pl.LightningModule):
    DETECTION_EMBEDDING_SIZE = 2048
    OUTPUT_EMBEDDING_SIZE = 768
    
    def __init__(self, NUM_FRAMES_PER_STEP=5, MAX_DETECTIONS=20, max_epochs=100, lr=1e-4, batch_size=4):
        super().__init__()
        
        self.NUM_FRAMES_PER_STEP = NUM_FRAMES_PER_STEP
        self.MAX_DETECTIONS = MAX_DETECTIONS
        self.CANDIDATES = self.NUM_FRAMES_PER_STEP * self.MAX_DETECTIONS
        
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        
        self.save_hyperparameters()
        
    def forward(self, steps, features, boxes, entities, entity_count):
        BATCH_SIZE = len(entity_count)
        NUM_ACTIONS = len(entity_count[0])
        
        inputs = self.lxmert_tokenizer(
            steps,
            padding="longest",
            truncation=False,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        # Need to manually put these on GPU for some reason. Lightning doesn't detect it!
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
        
        V = vision_embeddings.reshape(BATCH_SIZE, NUM_ACTIONS, self.CANDIDATES, self.OUTPUT_EMBEDDING_SIZE)
        
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
    
    def training_step(self, batch, batch_idx):
        _, bboxes, features, steps, entities, entity_count, _, _ = batch
        loss_data, VG, RR = self(steps, features, bboxes, entities, entity_count)
        
        loss_ = compute_loss_batched(loss_data) / self.batch_size
        
        total, correct = compute_alignment_accuracy_batched(loss_data)
        accuracy = correct / total
        
        self.log('loss', loss_, on_step=True, on_epoch=True, prog_bar=True)
        self.log('accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss_, 'accuracy': accuracy}
    
    def validation_step(self, batch, batch_idx):
        _, bboxes, features, steps, entities, entity_count, _, _ = batch
        loss_data, VG, RR = self(steps, features, bboxes, entities, entity_count)
        
        loss_ = compute_loss_batched(loss_data) / self.batch_size
        
        total, correct = compute_alignment_accuracy_batched(loss_data)
        accuracy = correct / total
        
        self.log('loss', loss_, on_step=True, on_epoch=True, prog_bar=True)
        self.log('accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss_, 'accuracy': accuracy}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
                
        scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer, int(0.2 * self.hparams.max_epochs), self.hparams.max_epochs),
                     'name': 'learning_rate',
                     'interval': 'epoch',
                     'frequency': 1}
        
        return [optimizer], [scheduler]
    