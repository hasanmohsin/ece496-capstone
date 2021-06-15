import itertools
import torch
import einops
import torch.nn.functional as F

from transformers import LxmertModel, LxmertTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from loss import loss_RA_MIL

from model import remove_unused2
from model import get_ent_inds
from model import get_entity_embeddings

class LxmertVGHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            torch.nn.GELU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_labels),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class ModelFC(nn.Module):
    NULL = '[unused1]'
    ENTITY = '[unused2]'
    ACTION = '[unused3]'

    VISUAL_EMBEDDING_SIZE = 2048
    LINGUISTIC_EMBEDDING_SIZE = 768
    BBOX_SIZE = 4
    
    MAX_SENTENCE_SIZE = 25

    def __init__(self, device, NUM_FRAMES_PER_STEP=5, DETECTIONS_PER_FRAME=20):
        super(ModelFC, self).__init__()

        self.device = device

        self.NUM_FRAMES_PER_STEP = NUM_FRAMES_PER_STEP
        self.DETECTIONS_PER_FRAME = DETECTIONS_PER_FRAME
        self.DETECTIONS_PER_STEP = self.NUM_FRAMES_PER_STEP * self.DETECTIONS_PER_FRAME

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert_tokenizer.add_special_tokens({"additional_special_tokens": [self.NULL, self.ENTITY, self.ACTION]})
        self.lxmert_tokenizer.encode([self.NULL, self.ENTITY, self.ACTION])

        self.NULL_TOKEN = self.lxmert_tokenizer.convert_tokens_to_ids(self.NULL)
        self.ENTITY_TOKEN = self.lxmert_tokenizer.convert_tokens_to_ids(self.ENTITY)
        self.ACTION_TOKEN = self.lxmert_tokenizer.convert_tokens_to_ids(self.ACTION)

        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert.to(device)
        
        self.VG = LxmertVGHead(self.lxmert.config, self.DETECTIONS_PER_STEP)
        self.VG.to(device)
        
    def forward(self, BATCH_SIZE, NUM_ACTIONS, steps, features, boxes, entity_count, entity_list):
        steps = remove_unused2(steps)
        
        entity_idx = get_ent_inds(self, entity_list, steps)[0]
        entity_idx = [entity_indices[-1] for entity_indices in entity_idx]
        
        entity_count = entity_count[0]

        features = features.to(self.device)
        features = features.reshape(1, (NUM_ACTIONS - 1), (self.DETECTIONS_PER_STEP), self.VISUAL_EMBEDDING_SIZE)
        features = features.squeeze()
        
        boxes = boxes.to(self.device)
        boxes = boxes.reshape(1, (NUM_ACTIONS - 1), (self.DETECTIONS_PER_STEP), self.BBOX_SIZE)
        boxes = boxes.squeeze()
    
        steps = steps * (NUM_ACTIONS - 1)
        
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
        
        E = output["language_output"][: , entity_idx, :]
        
        VG_logits = self.VG(E.reshape(-1, self.LINGUISTIC_EMBEDDING_SIZE))
        VG_logits = VG_logits.reshape((NUM_ACTIONS - 1), -1, self.DETECTIONS_PER_STEP)
        VG_logits = VG_logits.permute(1, 0, 2)
        
        idx_1 = torch.arange(VG_logits.shape[0])
        idx_2 = torch.arange(VG_logits.shape[1]).repeat_interleave(torch.tensor(entity_count), 0)
        
        VG_scores, VG_scores_index = VG_logits[idx_1, idx_2].softmax(dim=-1).max(dim=-1)
        
        scores = torch.zeros(((NUM_ACTIONS - 1), (NUM_ACTIONS - 1), max(entity_count), max(entity_count))).to(self.device)
        minimum_logit = VG_logits.min()
        
        e = -1
        b = 0

        for m in range(VG_logits.shape[1]):
            for j in range(max(entity_count)):
                # Detect padding.
                if not j < entity_count[m]:
                    scores[:, m, j, :] = minimum_logit - 1
                    continue
                e = e + 1
                b = -1
                for l in range(VG_logits.shape[1]):
                    for k in range(max(entity_count)):
                        # Detect padding.
                        if not k < entity_count[l]:
                            scores[l, m, j, k] = minimum_logit - 1
                            continue
                        bbox = b + 1
                        scores[l, m, j, k] = VG_logits[e, l, VG_scores_index[b]]
                        
        Y = torch.ones(BATCH_SIZE, (NUM_ACTIONS - 1), (NUM_ACTIONS), dtype=torch.float).cuda()
        VG_scores_index = pad_sequence(VG_scores_index.split(entity_count), batch_first=True)
        scores = scores.unsqueeze(0)
        
        return scores, Y, BATCH_SIZE, (NUM_ACTIONS - 1), None, None, VG_scores_index, None