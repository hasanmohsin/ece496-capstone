import spacy
import neuralcoref

from ling_obj_classes import *
 
#helper function for generators
#returns first value of generator if exists, or None
def get_first(generator):
    try:
        val = next(generator)
    except StopIteration:
        val = None
    return val


class ReferenceResolver:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

        self.coref = neuralcoref.NeuralCoref(self.nlp.vocab)
        self.nlp.add_pipe(self.coref, name ='neuralcoref')
        return

    #parses and finds coreferences
    def parse_and_ref_res(self, sentence):
        #parses and finds coreferences
        doc = self.nlp(sentence)

        all_main_phrases = [[]]
        all_indirect_words = [[]]

        i = 0

        for cluster in doc._.coref_clusters:
            
            print("\n\nCluster {}".format(i))

            main_phrase = cluster.main.text
            print("\nMain phrase: {}".format(main_phrase))

            all_main_phrases.append(main_phrase)

            indirect_words = []
            for mention in cluster.mentions:
                if mention.text != main_phrase:
                    indirect_words+=mention.text
                print("'{}' refers to this main phrase.".format(mention.text))
            
            all_indirect_words.append([indirect_words])

            i+=1

        all_main_phrases.pop(0)
        all_indirect_words.pop(0)

        return all_main_phrases, all_indirect_words

    
    def expand_refs(self, sentence):
        #parses and finds coreferences
        doc = self.nlp(sentence)

        new_str = ''

        #go over each word
        for token in doc:
            new_str+=token.text_with_ws

            if token._.in_coref:
                for cluster in token._.coref_clusters:
                    
                    main = cluster.main
                    
                    if token.text not in main.text:
                        new_str+=' (refers to \'{}\')'.format(main)
        return new_str

    #given a list of strings for step descriptions,
    #creates dictionary, where key = word, value = list of references for that word
    def dict_of_refs(self, step_text_list):
        sentence = ", then ".join(step_txt_list)
        
        #parses and finds coreferences
        doc = self.nlp(sentence)

        ref_dict = {}

        #go over each word
        for token in doc:

            ref_dict[token.text] = []

            if token._.in_coref:
                for cluster in token._.coref_clusters:

                    main = cluster.main
                    
                    

                    if token.text not in main.text:
                        ref_dict[token.text].append(main)
        return ref_dict

    #given a sentence, finds action_entity breakdown
    #sentence should refer to a specific step
    #returns dictionary
    #keys are the action word, value associated is the list of entities
    #related to that action
    def parse_action_entity(self, sentence):
        #first find the entities
        doc = self.nlp(sentence)

        noun_chunks = [chunk.text for chunk in doc.noun_chunks]

        print(noun_chunks)

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
                        if noun not in act_ent_dict[action]:
                            act_ent_dict[action].append(noun)
        """
        #try to find entities related to this action
        for action in actions:

            obj_ent = []

            possible_ents = [child for child in action.children]

            #for this action token, each child path is an entity
            for ents in possible_ents:
                obj = None

                #print("\nAt child {}".format(child.text))

                #first check if this child token is in some noun chunk: if so, 
                #add the entire noun_chunk as an object
                if any([(ents.text in chunk) for chunk in noun_chunks]):
                    
                    for noun_chunk in noun_chunks:
                        if ents.text in noun_chunk:
                            obj = noun_chunk
                            #print(obj)
                            break
                #we explore 
                #its children depth first and add the noun phrases they are in
                else:
                    #append its children to possible ents
                    for next_child in ents.children:
                        possible_ents.append(next_child)
                
                #put into object dictionary
                if obj:
                    obj_ent.append(obj)
            act_ent_dict[action.text] = obj_ent
        """

        return act_ent_dict

        #given a list of strings for step descriptions,

        #given a sentence, finds action_entity breakdown

    #sentence should refer to a specific step
    #returns list of action step objects
    # each has a pred (PRED), list of direct objects (DOBJ) (Entities), and list of propositional phrases (PP) (entities) in dictionary
    #keys are the action word, value associated is the list of entities
    #related to that action
    def parse_step(self, step_sent):
        #first find the entities
        doc = self.nlp(step_sent)

        noun_chunks = [chunk.text for chunk in doc.noun_chunks]

        #parse the sentence into predicates, and object arrays related to them
        #each action step will get 1 predicate, a list of dobj, and a list of pp
        #we return the list of action steps for the object
        #NOTE: in future, may restrict to 1 action step per instructional step

        #start dict with key = pred text, value = action_step
        action_step_dict = {}
        
        #add any pps
        for token in doc:
            if token.pos_ == "ADP" or token.dep_ == "dobj":
                #keep iterating up until you get action that is a 'VERB'
                action = token.head

                #we stop at first verb up or until we get to root 
                while action.pos_ != "VERB" and action.head != action:
                    #print(action)
                    action = action.head

                #get substring corresponding to pp or dobj
                ent_span = doc[token.left_edge.i : token.right_edge.i+1]

                if action.text not in action_step_dict.keys():
                        #create action step for it
                        action_step = ActionStep(pred = action.text)
                        action_step_dict[action.text] = action_step
                
                #add if not already in action_step_dict for this action
                if not ent_span in action_step_dict[action.text]:
                    if token.pos_ == "ADP":
                        action_step_dict[action.text].addPP(Entity(ent_span, "PP"))
                    elif token.dep_ == "dobj":
                        action_step_dict[action.text].addDOBJ(Entity(ent_span, "DOBJ"))

        """
        #then add any remaining direct objects (not corresponding to pp)
        for token in doc:
                #if a the token is a direct object
                if token.dep_ == "dobj":
                    #keep iterating up until you get action that is a 'VERB'
                    action = token.head

                    #we stop at first verb up or until we get to root 
                    while action.pos_ != "VERB" and action.head != action:
                        #print(action)
                        action = action.head
                    
                    #create action_step for this verb
                    #right now don't assign id, will do later when in order
                    #action_step = ActionStep(act_id = -1, action)

                    #get substring corresponding to dobj
                    dobj_span = doc[token.left_edge.i : token.right_edge.i+1]

                    #check if not seen this action before
                    if action not in action_step_dict.keys():
                        #create action step for it
                        action_step = ActionStep(pred = action)
                        action_step_dict[action] = action_step
                    
                    #add if not already in action_step_dict for this action
                    if not dobj_span in action_step_dict[action]:
                        action_step_dict[action].addDOBJ(Entity(dobj_span, "DOBJ"))

                    #find noun chunk this token is in
                    #for noun in doc.noun_chunks:
                    #    if token in noun:
                    #        #add the noun chunk the token is in to the action_step DOBJ list, if not already added
                    #        #(as part of DOBJ or PP lists)
                    #        if not noun in action_step_dict[action]:
                    #            action_step_dict[action].addDOBJ(Entity(noun, "DOBJ"))
        """

        return action_step_dict

    #sent is a sentence over multiple steps for video
    #corrects the action ids in action_step_dict
    def order_ids(self, sent, action_step_dict):
        doc = self.nlp(sent)

        print(action_step_dict.keys())

        count = 0
        for token in doc:
            #print("At token: {}".format(token.text))
            if token.text in action_step_dict.keys():
                #print("At count : {}".format(count))
                action_step_dict[token.text].setActId(count)
                count +=1

        return action_step_dict

    #returns pred (text) and action ID the given token is in
    #"" and -1 if not found
    def find_action_step_for_token(self, token, action_step_dict):
        for pred_key in action_step_dict.keys():
            #the token is a predicate for this action step
            if action_step == pred_key:
                return pred_key, action_step_dict[pred_key].act_id


            if action_step_dict[pred_key].hasTokenEntity(token):
                return pred_key, action_step_dict[pred_key].act_id

        return "", -1

    #creates dictionary, where key = entity, value = list of actions entity refers to
    def resolve_refs_after_parse(self, step_text_list, action_step_dict):

        #next do coreference resolution
        sentence = ", then ".join(step_text_list)
        
        #parses and finds coreferences
        doc = self.nlp(sentence)

        #go over each word
        for token in doc:

            if token._.in_coref:
                for cluster in token._.coref_clusters:

                    main = cluster.main
                    
                    

                    if token not in main:
                        #main the token this one refers to

                        #find action step token is in, 
                        #find the action step main is in, and set the reference id
                        pred_token, token_action_id = self.find_action_step_for_token(token, action_step_dict)
                        pred_ref, ref_action_id = self.find_action_step_for_token(main, action_step_dict)
                        
                        #looks at entities corresponding to this token for this action step, and inserts the
                        #action ID they refer to for them
                        if pred_token != "":
                            action_step_dict[pred_token].set_references(token, ref_action_id)

                        #ref_dict[token].append(main)
        return action_step_dict

        #main run method
        #pass a list of steps (strings)
        #this will parse them all, and resolve references
        #returning dictionary, key = predicate text, value = ActionStep object
        #def parse_and_resolve_all_refs(self, step_list):
            
        #    for step in step_list:

rr_model = ReferenceResolver()

eg_text_1 = "Grab the spatula and use it to mix it with the dried onions." 
eg_text_2 = "drizzle little bit of olive oil on both the sides of 4 bread slices"

#act_ent_dict =rr_model.parse_action_entity(eg_text_2)

#print(act_ent_dict.keys())
#print(act_ent_dict.values())

#NOTE: action step pred is a string, but for Entities, the ent_text is a Spacy object. use .text to get actual
#string
action_step_dict = rr_model.parse_step(eg_text_2)
action_step_dict = rr_model.order_ids(eg_text_2, action_step_dict)

#takes in parsed list of ordered action steps, and finds references
# puts reference id in entity ref_id field 
action_step_dict = rr_model.resolve_refs_after_parse(eg_text_2, action_step_dict)

action_steps = action_step_dict.values()

for action_step in action_steps:
    print("\nAction ID: {}".format(action_step.act_id))
    print("PRED: {}".format(action_step.pred))

    dobj_list_str = [(dobj.ent_text.text + " (Refers to action ID: {})".format(dobj.act_id_ref)) for dobj in action_step.dobj_list]
    pp_list_str = [(pp.ent_text.text + " (Refers to action ID: {})".format(pp.act_id_ref)) for pp in action_step.pp_list]

    print("DOBJ: {}".format(dobj_list_str))
    print("PP: {}".format(pp_list_str))