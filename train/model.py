# https://github.com/google-research/bert/issues/635
# https://colab.research.google.com/drive/18TyuMfZYlgQ_nXo-tr8LCnzUaoX0KS-h?usp=sharing#scrollTo=W4cZIVrg82ua.

import itertools
import torch
import einops

from transformers import LxmertModel, LxmertTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from loss import loss_RA_MIL

class Model(nn.Module):
    NULL = '[unused1]'
    ENTITY = '[unused2]'
    ACTION = '[unused3]'

    DETECTION_EMBEDDING_SIZE = 2048
    OUTPUT_EMBEDDING_SIZE = 768

    def __init__(self, device, NUM_FRAMES_PER_STEP=5, MAX_DETECTIONS=20):
        super(Model, self).__init__()

        self.device = device

        self.NUM_FRAMES_PER_STEP = NUM_FRAMES_PER_STEP
        self.MAX_DETECTIONS = MAX_DETECTIONS

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert_tokenizer.add_special_tokens({"additional_special_tokens": [self.NULL, self.ENTITY, self.ACTION]})
        self.lxmert_tokenizer.encode([self.NULL, self.ENTITY, self.ACTION])

        self.NULL_TOKEN = self.lxmert_tokenizer.convert_tokens_to_ids(self.NULL)
        self.ENTITY_TOKEN = self.lxmert_tokenizer.convert_tokens_to_ids(self.ENTITY)
        self.ACTION_TOKEN = self.lxmert_tokenizer.convert_tokens_to_ids(self.ACTION)

        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert.to(device)

        self.null = torch.nn.Linear(self.OUTPUT_EMBEDDING_SIZE, NUM_FRAMES_PER_STEP * MAX_DETECTIONS * self.OUTPUT_EMBEDDING_SIZE)
        self.null.to(device)

    def forward(self, BATCH_SIZE, NUM_ACTIONS, steps, features, boxes, entity_count):
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

        token_ids = inputs.input_ids

        entity_idx = ((token_ids == self.ENTITY_TOKEN) | (token_ids == self.NULL_TOKEN))
        action_idx = (token_ids == self.ACTION_TOKEN)
        null_idx = (token_ids == self.NULL_TOKEN)

        entity_embeddings = output['language_output'][entity_idx]
        action_embeddings = output['language_output'][action_idx]
        vision_embeddings = output['vision_output']
        null_embedding = output['language_output'][null_idx]

        split_sizes = torch.tensor(entity_count).flatten().tolist()
        entities = entity_embeddings.split(split_sizes)

        E = pad_sequence(entities, batch_first=True)
        max_entities = E.shape[1]
        E = E.reshape(-1, NUM_ACTIONS, E.shape[1], E.shape[2])

        A = action_embeddings.reshape(BATCH_SIZE, NUM_ACTIONS, -1)
        V = vision_embeddings.reshape(BATCH_SIZE, NUM_ACTIONS - 1, CANDIDATES, -1)

        # Add generated null features to the outputted vision features.
        null_features = self.null(null_embedding).reshape(BATCH_SIZE, self.NUM_FRAMES_PER_STEP * self.MAX_DETECTIONS,  self.OUTPUT_EMBEDDING_SIZE)
        V = torch.cat((V, null_features.unsqueeze(1)), 1)

        # Calculate RR (RR_scores_index).
        #note: we write c here as number of actions as well
        #since we don't wish to sum over a
        RR_scores = torch.einsum('baes, bcs -> baec', E, A)

        edge_mask = torch.ones(NUM_ACTIONS, NUM_ACTIONS).tril(diagonal=-1).cuda()
        edge_mask[-1, :] = 0
        edge_mask[:, -1] = 1
        edge_mask = einops.repeat(edge_mask, 'x y -> b x c y', b=BATCH_SIZE, c=max_entities)

        RR_scores_max, RR_scores_index = (RR_scores * edge_mask).max(dim=-1)

        # Update entities to include actions.
        RR_combine = RR_scores_index.flatten(1, 2)
        action_index = torch.arange(RR_combine.shape[0]).repeat_interleave(RR_combine.shape[1]).reshape(RR_combine.shape[0], -1)
        A_spread = A[action_index, RR_combine].reshape(E.shape)

        E = E + A_spread

        # Calculate loss_E.
        loss_E = E

        # Calculate VG (VG_scores_index) and loss_V.
        VG_scores = torch.einsum('bacs, baes -> baec', V, E)
        VG_scores_max, VG_scores_index = VG_scores.max(dim=-1)

        V_flat = V.reshape(-1, self.OUTPUT_EMBEDDING_SIZE)

        VG_scores_index_flat = VG_scores_index.flatten()
        offsets = torch.arange(0, BATCH_SIZE * NUM_ACTIONS * CANDIDATES, CANDIDATES).cuda()
        offsets = offsets.repeat_interleave(max_entities)

        VG_scores_index_flat = VG_scores_index_flat + offsets

        loss_V = V_flat[VG_scores_index_flat, :].reshape(BATCH_SIZE, NUM_ACTIONS, max_entities, -1)

        # Calculate loss_R.
        loss_R = torch.ones((BATCH_SIZE, NUM_ACTIONS, NUM_ACTIONS)).cuda()

        dim_1 = RR_scores_index.reshape(BATCH_SIZE, -1)
        dim_2 = torch.arange(NUM_ACTIONS).repeat_interleave(max_entities)
        dim_2 = einops.repeat(dim_2, 'd -> b d', b=BATCH_SIZE)

        loss_R[:, dim_1, dim_2] = 0.0

        #entity embeddings, selected visual grounding embeddings, adjacency list for
        #ref resolution
        return loss_E, loss_V, loss_R, VG_scores_index, RR_scores_index, A, E, V
