import torch


def get_alignment_accuracy(model, dataloaders, across_all=False):
    total = 0
    correct = 0
        
    with torch.no_grad():
        for dataloader in dataloaders:
            for data in dataloader:
                _, boxes, features, _, steps, entities, entity_count, _ = data
                loss_data, VG, RR = model(steps, features, boxes, entities, entity_count)

                _total, _correct = compute_alignment_accuracy_batched(loss_data, across_all)

                total = total + _total
                correct = correct + _correct
        
    return (correct / total)

def compute_alignment_accuracy_batched(loss_data, across_all):
    total = 0
    correct = 0
    
    alignment_scores, entity_count, BATCH_SIZE, NUM_ACTIONS, MAX_ENTITIES = loss_data
    
    for batch_idx in range(BATCH_SIZE):
        _alignment_scores = alignment_scores[batch_idx]
        _entity_count = entity_count[batch_idx]
        
        if across_all:
            _total, _correct = compute_alignment_accuracy_all((_alignment_scores, _entity_count, NUM_ACTIONS, MAX_ENTITIES))
        else:
            _total, _correct = compute_alignment_accuracy((_alignment_scores, _entity_count, NUM_ACTIONS, MAX_ENTITIES))
        
        total = total + _total
        correct = correct + _correct
    
    return total, correct

def compute_alignment_accuracy(loss_data):
    total = 0
    correct = 0
    
    alignment_scores, entity_count, NUM_ACTIONS, MAX_ENTITIES = loss_data
    
    # l: ENTITY_ACTION_ID
    # e: ENTITY_ID
    # m: CANDIDATE_ACTION_ID
    
    for m in range(NUM_ACTIONS):
        for e in range(entity_count[m]):
            for l in range(NUM_ACTIONS):
                if m == l:
                    continue
                    
                aligned = compute_alignment(alignment_scores[m, e, m, :], alignment_scores[m, e, l, :])
                
                if aligned:
                    correct = correct + 1
                    
                total = total + 1
                
    return total, correct

def compute_alignment_accuracy_all(loss_data):
    total = 0
    correct = 0
    
    alignment_scores, entity_count, NUM_ACTIONS, MAX_ENTITIES = loss_data
    
    # l: ENTITY_ACTION_ID
    # e: ENTITY_ID
    # m: CANDIDATE_ACTION_ID
    
    for m in range(NUM_ACTIONS):
        for e in range(entity_count[m]):
            aligned_across_all = True
            for l in range(NUM_ACTIONS):
                if m == l:
                    continue
                    
                aligned = compute_alignment(alignment_scores[m, e, m, :], alignment_scores[m, e, l, :])
                
                if not aligned:
                    aligned_across_all = False
                    break
            
            if aligned_across_all:
                correct = correct + 1
            
            total = total + 1
                
    return total, correct
    
def compute_alignment(score_m, score_l):
    '''
        score_m: score between entity from STEP M and candidates from STEP M
        score_l: score between entity from STEP M and candidates from STEP L
        
    '''
    
    m_max = score_m.max()
    l_max = score_l.max()
    
    return (m_max > l_max)