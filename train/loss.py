import torch


def get_alignment_loss(model, dataloaders, margin=10):
    loss = 0    
    datapoints = 0
        
    with torch.no_grad():
        for dataloader in dataloaders:
            for data in dataloader:
                _, boxes, features, actions, steps, entities, entity_count, _ = data
                loss_data, VG, RR = model(steps, features, boxes, entities, entity_count)

                loss = loss + compute_loss_batched(loss_data, margin)
                datapoints += len(steps) * len(actions[0])
    
    return loss / datapoints

def compute_loss_batched(loss_data, margin=10):
    loss = 0
    
    alignment_scores, entity_count, BATCH_SIZE, NUM_ACTIONS, MAX_ENTITIES = loss_data
    
    for batch_idx in range(BATCH_SIZE):
        _alignment_scores = alignment_scores[batch_idx]
        _entity_count = entity_count[batch_idx]
        
        loss = loss + compute_loss((_alignment_scores, _entity_count, NUM_ACTIONS, MAX_ENTITIES), margin)
        
    return loss

def compute_loss(loss_data, margin=0):    
    alignment_scores, entity_count, NUM_ACTIONS, MAX_ENTITIES = loss_data
    
    # Recall the shape of the alignment scores tensor:
    # ENTITY_ACTION_ID * ENTITY * CANDIDATE_ACTION_ID * CANDIDATE.
    
    S = torch.zeros((NUM_ACTIONS, NUM_ACTIONS))
    zero = torch.zeros((1, 1))
    
    # l: ENTITY_ACTION_ID
    # m: CANDIDATE_ACTION_ID
    
    for l in range(NUM_ACTIONS):
        for m in range(NUM_ACTIONS):
            S[l][m] = compute_S(alignment_scores[m, :, l, :], entity_count[m])
            
    loss = 0
    
    for l in range(NUM_ACTIONS):
        for m in range(NUM_ACTIONS):
            if l == m:
                continue
            loss = loss + torch.max(S[l][m] - S[l][l] + margin, zero) + torch.max(S[m][l] - S[l][l] + margin, zero)
            
    return loss
        
def compute_S(scores, entity_count):
    '''
        scores: Alignment scores between entities from STEP M and candidates (boxes) from step STEP L.
    '''
    
    if entity_count == 0:
        return 0.
    
    # Remove padded dimension.
    scores = scores[:entity_count]
    
    S = scores.max(dim=-1)[0].sum()
    
    return S