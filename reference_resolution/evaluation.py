import numpy as np
import spacy

from difflib import SequenceMatcher
from numpy import unravel_index
from ref_res_model import nlp

threshold = 0.8

class Stat:
    def __init__(self, tp, tn, fp, fn, fpp, fnp, pm):
        self.true_positive = tp
        self.true_negative = tn
        self.false_positive = fp
        self.false_negative = fn
        self.false_positive_parse = fpp
        self.false_negative_parse = fnp
        self.pred_mismatch = pm

def root_nouns(s):
    doc = nlp(s.text)
    return [noun.root.text for noun in doc.noun_chunks]

def is_similar(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def compare_entities(p_entities, g_entities):
    pair_score = np.zeros((len(p_entities), len(g_entities)))
    pairs = []

    for i, p_entity in enumerate(p_entities):
        for j, g_entity in enumerate(g_entities):
            pair_score[i][j] = compare_entity(p_entity.get_text(), g_entity.get_text())


    for iter in range(len(g_entities)):
        indices = unravel_index(pair_score.argmax(), pair_score.shape)
        i, j = indices[0], indices[1]

        if np.amax(pair_score) > threshold:
            pairs.append((p_entities[i], g_entities[j]))
            pair_score[i, :] = 0
            pair_score[:, j] = 0

    return pairs


def compare_entity(p_text, g_text):
    p_root_nouns = root_nouns(p_text)
    g_root_nouns = root_nouns(g_text)

    #print("p_text: {}".format(p_text))
    #print("g_text: {}".format(g_text))

    #print(len(p_root_nouns))
    #print(len(g_root_nouns))

    if len(p_root_nouns) == 0:
        p_root_nouns.append(p_text)

    if len(g_root_nouns) == 0:
        g_root_nouns.append(g_text)
    
    pair_scores = np.zeros((len(p_root_nouns), len(g_root_nouns)))

    for i, p_root_noun in enumerate(p_root_nouns):
        for j, g_root_noun in enumerate(g_root_nouns):
            pair_scores[i,j] = is_similar(g_root_noun, p_root_noun)

    return np.amax(pair_scores)


def evaluate(p_steps, g_steps):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    false_positive_parse = 0
    false_negative_parse = 0

    pred_mismatch = 0

    for p_action_step, g_action_step in zip(p_steps, g_steps):
        p_unmatched = set(p_action_step.dobj_list + p_action_step.pp_list)
        g_unmatched = set(g_action_step.dobj_list + g_action_step.pp_list)

        if is_similar(p_action_step.pred, g_action_step.pred):
            pairs_1 = compare_entities(p_action_step.dobj_list, g_action_step.dobj_list)
            pairs_2 = compare_entities(p_action_step.pp_list, g_action_step.pp_list)

            for pair in (pairs_1 + pairs_2):
                p_entity = pair[0]
                g_entity = pair[1]

                p_unmatched = p_unmatched - {p_entity}
                g_unmatched = g_unmatched - {g_entity}

                if p_entity.act_id_ref == g_entity.act_id_ref and (p_entity.act_id_ref != -1 and g_entity.act_id_ref != -1):
                    true_positive += 1
                elif p_entity.act_id_ref != -1 and g_entity.act_id_ref == -1:
                    false_positive += 1
                elif p_entity.act_id_ref == -1 and g_entity.act_id_ref != -1:
                    false_negative += 1
                elif p_entity.act_id_ref == -1 and g_entity.act_id_ref == -1:
                    true_negative += 1
        else:
            pred_mismatch = pred_mismatch + 1

        for entity in p_unmatched:
            if entity.act_id_ref == -1:
                false_negative_parse += 1
            else:
                false_positive_parse += 1

        for entity in g_unmatched:
            if entity.act_id_ref == -1:
                false_positive_parse += 1
            else:
                false_negative_parse += 1

    return Stat(true_positive, true_negative, false_positive, false_negative, false_positive_parse, false_negative_parse, pred_mismatch)

        