import torch

def loss_RA_MIL(y, R, E, V):
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
            The size of the matrix is M x M. M is the number of steps.
            
            R_lj contains a 1 if there is any backward reference between a_l and a_j.
            
        V:
            visual embeddings
            -----------------------
            The size of the matrix is M x J x 768. M is the number of steps and J is the maximum
            number of entities across all of the steps.
            
            V_mj contains the visual embedding that corresponds to e_mj entity. This is the
            b_mj bounding box (bounding box with highest alignment score). Since each the
            steps may have a different number of entities.
            
        E:
            reference aware entity embeddings
            -----------------------
            The size of the matrix is M x J x 768. E_mj contains the reference aware entity embedding
            of e_mj. Since each the steps may have a different number of entities.
    '''
    
    M = E.shape[0]
    J = E.shape[1]

    # Compute the outer product between E and V. This essentially calculates
    # all possible alignment scores across the entire video. scores_lmjk is
    # the alignment score of e_mj and b_lk (same as equation). The size of
    # the matrix is M x J x M x J.
    #
    # Ref: stackoverflow.com/questions/24839481/python-matrix-outer-product
    #
    # Note that the alignment score for the padding matrices will be -inf due
    # to the way they're configured (1 * -inf).
    scores = torch.einsum('mjd, lkd -> lmjk', E, V)
    #scores[scores.isnan()] = float('-inf')

    # The best alignment score between e_mj and b_lk (over all k).
    max_k_align = scores.max(3)[0]
    #max_k_align[max_k_align == float('-inf')] = 0

    # Sum all of the best alignment scores from e_mj (over all j).
    S_lm = max_k_align.sum(2)

    # Find the transposed version.
    S_ml = S_lm.transpose(0, 1)

    # Compute reference based penalty. 1 if none of the entities in e_m refer
    # to a_l, constant (hyperparameter) otherwise. We can use R since it has
    # the mappings between each of the actions. This is a M x M matrix.
    Y_lm = R * y
    Y_ml = Y_lm.transpose(0, 1)

    # Zero matrix.
    zero = torch.zeros(M, M)

    # S_ll needs to have the rows filled with diagonal values. Note that unsqueeze(1)
    # for a vector is the same as transposing it.
    S_ll = S_lm.diagonal().unsqueeze(1).repeat(1, M)

    # Vectorization magic.
    loss = (Y_lm * torch.max(zero, S_lm - S_ll) + Y_ml * torch.max(zero, S_ml - S_ll)).sum()
    
    return loss