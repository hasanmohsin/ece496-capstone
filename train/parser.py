import spacy
import neuralcoref

nlp = spacy.load('en_core_web_sm')

def parse(steps, max_step_length=None):
    entities = []

    for step in steps:
        ents = list(get_entities(step).values())
        if len(ents) == 0:
            entities.append([])
        else:
            entities.append(ents[0])

    indices = []

    for idx, data in enumerate(zip(entities, steps)):
        es = data[0]
        step = data[1]
        start = 0

        if len(es) == 0:
            indices.append([])
        else:
            for e in es:
                if max_step_length:
                    shift = idx * max_step_length
                else:
                    shift = 0

                mapping_entity, start = get_index(step, e, start, shift)
                indices.append(mapping_entity)

    return entities, indices

def get_index(sentence, entity, start=0, shift=0):
    mapping = []
    words_sentence = sentence.split()
    words_entity = entity.split()
    words_count = len(words_entity)

    e = words_entity.pop(0)

    for idx, word in enumerate(words_sentence[start:]):
        if word == e:
            mapping.append(idx + start + shift)
            if len(words_entity) > 0:
                e = words_entity.pop(0)

        if len(mapping) == words_count:
            break

    if len(mapping) == 0:
        return mapping, (start + 1 - shift)
    else:
        return mapping, (mapping[-1] + 1 - shift)

def get_entities(sentence):
    doc = nlp(sentence)

    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    #parse the sentence into actions and object arrays
    #action is the single action of the sentence, obj_ent is a list of entities (objects of the action)
    actions = []
    act_ent_dict = {}

    #first identify the verbs with objects
    for token in doc:
        noun_token = None

        #identify the noun chunk the token is in, and get the root of the
        for noun in doc.noun_chunks:
            if token in noun:
                noun_token = noun.root
            
                if noun_token.dep_ == "iobj" or noun_token.dep_ == "dobj" or noun_token.dep_ == "pobj":
                    #keep iterating up until you get action that is a 'VERB'
                    action = noun_token.head

                    #we stop at first verb up or until we get to root 
                    while action.pos_ != "VERB" and action.head.text != action.text:
                        #print(action)
                        action = action.head

                    #if action

                    actions.append(action)

                    #if haven't added this action yet
                    if action not in act_ent_dict.keys():
                        act_ent_dict[action] = []

                    #append the noun chunk the token is in
                    if noun.text not in act_ent_dict[action]:
                        act_ent_dict[action].append(noun.text)

    return act_ent_dict
