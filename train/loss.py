import torch
import torch.nn.functional as F

def loss_RA_MIL(y=1, B=None, M=None, VG_dist1=None, VG_dist2=None, scores=None, Y=None):
    '''
    Custom loss function for reference-aware visual grounding. This function must
    use PyTorch functions for any parameters that are to be trained. Some of the
    embeddings have been modified to match the VisualBERT model.
    
    The outputs of the VG extension are the alignment scores between the bounding
    box proposals and entities for that particular step. The size of this matrix
    is E x P, where E is the number of entities and P is the number of proposals.
    
    Source: http://vision.stanford.edu/pdf/huang-buch-2018cvpr.
    
    Input:
        y: alignment penalty (hyperparameter)
    
        R:
            references
            -----------------------
            The size of the matrix is BxM x M. M is the number of (action) steps.
            
            R_lj contains a 1 if there is any backward reference between a_l and a_j.
            
        V:
            visual embeddings
            -----------------------
            The size of the matrix is B x M x J x 768. M is the number of steps and J is the maximum
            number of entities across all of the steps.
            
            V_mj contains the visual embedding that corresponds to e_mj entity. This is the
            b_mj bounding box (bounding box with highest alignment score). Since each the
            steps may have a different number of entities.
            
        E:
            reference aware entity embeddings
            -----------------------
            The size of the matrix is B x M x J x 768. E_mj contains the reference aware entity embedding
            of e_mj. Since each the steps may have a different number of entities.
    '''        
    # The best alignment score between e_mj and b_lk (over all k).
    max_k_align = scores.max(dim=-1)[0]
    
    # Sum all of the best alignment scores from e_mj (over all j).
    S_lm = max_k_align.sum(dim=-1)

    # Find the transposed version.
    S_ml = S_lm.transpose(1, 2)

    # Compute reference based penalty. 1 if none of the entities in e_m refer
    # to a_l, constant (hyperparameter) otherwise. We can use R since it has
    # the mappings between each of the actions. This is a M x M matrix.
    Y_lm = torch.clamp(Y[:,:,:-1] + y, 0.0, 1.0)
    Y_ml = Y_lm.transpose(1, 2)
        
    # Zero matrix.
    zero = torch.zeros(B, M, M, dtype=torch.float).cuda()

    # S_ll needs to have the rows filled with diagonal values. Note that unsqueeze(1)
    # for a vector is the same as transposing it.
    S_ll = S_lm.diagonal(dim1=1, dim2=2).unsqueeze(2).repeat(1, 1, M)
    
    # Vectorization magic.
    loss_alignment = (Y_lm * torch.max(zero, S_lm - S_ll) + Y_ml * torch.max(zero, S_ml - S_ll)).sum()
    
    loss = loss_alignment
    
    # Include KL divergence.
    if VG_dist1 is not None and VG_dist2 is not None:
        target = B * 250
        loss_KL_div = F.kl_div(VG_dist1, VG_dist2, log_target=True, reduction='sum')
        loss_KL_div_eff = target - torch.clamp(loss_KL_div, max=target)
        loss = loss + loss_KL_div_eff
        print("KL_Div Loss: {}".format(loss_KL_div)) 

    print("Alignment Loss: {}".format(loss_alignment))

    return loss