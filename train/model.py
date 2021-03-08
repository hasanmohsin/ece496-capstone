# https://colab.research.google.com/drive/18TyuMfZYlgQ_nXo-tr8LCnzUaoX0KS-h?usp=sharing#scrollTo=W4cZIVrg82ua.

import itertools
import torch

from transformers import LxmertModel, LxmertTokenizer
from torch import nn

class Model(nn.Module):
    def __init__(self, max_step_length):
        '''
            steps: BATCH_SIZE * number of steps per video
            images: BATCH_SIZE * number of steps per video * number of candidates
            
            number of steps per video = number of images per video
        '''
        super(Model, self).__init__()
        self.max_step_length = max_step_length
        
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        
        self.softmax = nn.Softmax(dim=1)
    
    # Average the individual embedding of each entity.
    def get_entity_embeddings(self, output, entities, indices):
        # Flatten the first dimension (ie. number of steps) and indices.
        embeddings = output['language_output'].flatten(start_dim=0, end_dim=1)
        entity_embeddings = []
        
        for entity in indices:
            entity_embeddings.append(embeddings[entity].mean(axis=0)) 
        
        return torch.stack(entity_embeddings)
    
    # Perform reference resolution.
    def get_RR(self, steps, entities, action_embeddings, entity_embeddings):
        # Score calculated with dot product.
        RR = torch.einsum('mj, lj -> lm', action_embeddings, entity_embeddings)
        
        step_indices = list(itertools.chain.from_iterable([[idx] * len(es) for idx, es in enumerate(entities)]))

        # Prepare mask so that entities don't refer to steps in the future.
        # https://stackoverflow.com/questions/57548180/filling-torch-tensor-with-zeros-after-certain-index.
        mask = torch.zeros(RR.shape[0], RR.shape[1] + 1)
        mask[(torch.arange(RR.shape[0]), step_indices)] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        mask[:, -1] = 0
        RR = RR * (1. - mask[..., None]).squeeze(2)
        
        return self.softmax(RR), step_indices
    
    # Perform visual grounding.
    def get_VG(self, entities, steps, entity_embeddings, visual_embeddings):
        step_indices = torch.tensor([[idx, idx2] for idx, es in enumerate(entities) for idx2, e in enumerate(es)])
        max_entities = max([len(es) for es in entities])
        
        E_expanded = torch.zeros(len(steps), max_entities, visual_embeddings.shape[2])
        E_expanded[step_indices[:, 0], step_indices[:, 1]] = entity_embeddings
        scores, VG = torch.einsum('ijk, iyk -> ijyk', E_expanded, visual_embeddings).sum(dim=3).max(dim=2)
        
        E_expanded[scores == 0] = 0
        
        return VG, scores, E_expanded
    
    def get_loss_V(self, steps, entities, visual_embeddings, VG, VG_scores):
        max_entities = max([len(es) for es in entities])
        VG_features = visual_embeddings[torch.arange(0, len(steps)).repeat_interleave(3), VG.flatten()].reshape(len(steps), max_entities, -1)
        VG_features[VG_scores == 0] = 0
        
        return VG_features
        
    def forward(self, steps, boxes, features, entities, indices):
        inputs = self.lxmert_tokenizer(
            steps,
            padding="max_length",
            max_length=self.max_step_length,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        output = self.lxmert(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False
        )        
        
        action_embeddings = output['language_output'].mean(axis=1)
        entity_embeddings = self.get_entity_embeddings(output, entities, indices)
        visual_embeddings = output['vision_output']
        
        RR, step_indices = self.get_RR(steps, entities, action_embeddings, entity_embeddings)
        VG, VG_scores, loss_E = self.get_VG(entities, steps, entity_embeddings, visual_embeddings)
        
        loss_R = torch.zeros(RR.shape[1], RR.shape[1])
        loss_R[RR.argmax(dim=1), step_indices] = 1 
        
        loss_V = self.get_loss_V(steps, entities, visual_embeddings, VG, VG_scores)

        return output, RR, VG, VG_scores, loss_V, loss_E, loss_R, action_embeddings, entity_embeddings, visual_embeddings
