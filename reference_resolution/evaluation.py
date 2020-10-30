import numpy as np
import spacy

from difflib import SequenceMatcher
from numpy import unravel_index
from ref_res_model import nlp

threshold = 0.8

def root_nouns(s):
    doc = nlp(s)
    return [noun.root for noun in doc.noun_chunks]

def is_similiar(s1, s2)
    return SequenceMatcher(None, s1, s2).ratio()

def compare_entities(p_entities, g_entities):
    pair_scores = np.zeros((len(p_entities), len(g_entities)))
    pairs = []

    for i, p_entity in enumerate(p_entity):
        for j, g_entity in enumerate(g_entity):
            pair_score[i][j] = CompareEntity(p_entity.ent_text, g_entity.ent_text)

    for iterations(len(g_entities)):
        indices = unravel_index(pair_score.argmax(), pair_score.shape())
        i, j = indices[0], indices[1]

        if max(pair_score) > threshold:
            pairs.append((p_entities[i], g_entities[j]))
            pair_scores[i, :] = 0
            pair_scores[:, j] = 0

    return pairs


def compare_entity(p_text, g_text):
	p_root_nouns = root_nouns(p_text)
	g_root_nouns = root_nouns(g_text)

	pair_scores = np.zeros((len(p_root_nouns), len(g_root_nouns)))

	for i, p_root_noun in enumerate(p_root_nouns):
		for j, g_root_noun in enumerate(g_root_nouns):
			pair_scores[i][j] = is_similiar(g_root_noun, p_root_noun)

	return max(pair_score)


def evaluation(p_steps, g_steps):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    false_positive_parse = 0
    false_negative_parse = 0

    pred_mismatch = 0

    p_unmatched = set(p_steps.dobj_list + p_steps.pp_list)
    g_unmatched = set(g_steps.dobj_list + g_steps.pp_list)

    for p_action_step, g_action_step in zip(p_steps, g_steps):
        if similar(p_action_step.pred, g_action_step.pred):
            pairs_1 = CompareEntities(p_action_step.dobj_list, g_action_step.dobj_list)
            pairs_2 = CompareEntities(p_action_step.pp_list, g_action_step.pp_list)

            for pair in (P1 + P2):
                p_entity = pair[0]
                g_entity = pair[1]

                p_unmatched = p_unmatched - p_entity
                g_unmatched = g_unmatched - g_entity

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