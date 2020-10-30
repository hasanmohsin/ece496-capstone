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

        self.coref = neuralcoref.NeuralCoref(self.nlp.vocab, greedyness = 0.9, max_dist = 1000, max_dist_match = 1000)
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


        return act_ent_dict

        #given a list of strings for step descriptions,

        #given a sentence, finds action_entity breakdown

    #parse the step, but assumed only 1 action step 
    #for that step (eg. identifies only 1 predicate)
    #returns a single action step
    def parse_one_action_per_step(self, step_sent, step_num):
        doc = self.nlp(step_sent)

        action_step = None

        for token in doc:
            #get the root
            if token.dep_ == "ROOT":
                action = token

                #related spen
                action_span = doc[token.left_edge.i : token.right_edge.i+1]
                action_step = ActionStep(pred = action.text, act_id= step_num)
                
                #we assign all pp and dobj in step to this action
                for subtoken in action_span:
                    if subtoken.pos_ == "ADP" or subtoken.dep_ == "dobj":
                    
                        #get substring corresponding to pp or dobj
                        ent_span = doc[subtoken.left_edge.i : subtoken.right_edge.i+1]
                        
                        #if not already added
                        if not action_step.hasTokenEntity(subtoken): 
                            if subtoken.pos_ == "ADP":
                                action_step.addPP(Entity(ent_span, "PP"))
                            elif subtoken.dep_ == "dobj":
                                action_step.addDOBJ(Entity(ent_span, "DOBJ"))
                
                #we only extract 1 sentence
                break 
        return action_step
            

    #same as above, but could have multiple actions per step
    #sentence should refer to a specific step
    #returns list of action step objects
    # each has a pred (PRED), list of direct objects (DOBJ) (Entities), and list of propositional phrases (PP) (entities) in dictionary
    #keys are the action word, value associated is the list of entities
    #related to that action
    def parse_step_multiple_action_per_step(self, step_sent):
        #first find the entities
        doc = self.nlp(step_sent)

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
                while action.dep_ != "VERB" and action.head != action:
                    #print(action)
                    action = action.head

                #get substring corresponding to pp or dobj
                ent_span = doc[token.left_edge.i : token.right_edge.i+1]

                #want our action steps to be non-overlapping
                #check if the action is either already added

                #or if it is not part of an entity in the existing dictionary
                #if this  = True, it already in list
                #skip this action
                #NOTE: take this out to have multiple actions per step and allow overlapping (eg. an action step is a substep of another)
                if any([action_step.hasTokenEntity(action) for action_step in action_step_dict.values()]):
                    continue

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


        return action_step_dict

    #sent is a sentence over multiple steps for video
    #corrects the action ids in action_step_list
    def order_ids(self, sent, action_step_list):
        doc = self.nlp(sent)

        ordered_action_step_list = []

        #we take in an action step list, and produce another list
        #with the correct ordering, and with action IDs filled

        count = 0
        for token in doc:
            #print("At token: {}".format(token.text))
            #if the token text is part of the predicate
            for action_step in action_step_list:
                #each action step is labelled once, and no other action 
                #step gets the same value
                #only set if not set yet
                if token.text in action_step.pred and action_step.act_id == -1:
                    #print("At count : {}".format(count))
                    action_step.setActId(count)

                    ordered_action_step_list.append(action_step)
                    count +=1
                    break

        return ordered_action_step_list

    #sets ids assuming action_step_list is in correct order
    def order_ids_stepwise(self, action_step_list):
        
        count = 0
        for action_step in action_step_list:
           action_step.act_id = count
           count+=1
           
        return action_step_list

    #returns action ID the given token is in (as a pred, dobj or pp)
    #in most cases the token should be an entity (not a predicate)
    #-1 if not found
    def find_action_step_for_token(self, token, action_step_list):
        for action_step in action_step_list:
            #the token is a predicate for this action step
            if action_step.pred == token.text:
                return action_step.act_id


            if action_step.hasTokenEntity(token):
                return action_step.act_id

        return -1

    #given a list of strings, each corresponding to each step, produces 
    #a list of action_steps, ordered with action IDs
    def parse_step_list(self, step_text_list):
        action_step_list = []

        #parse each step individually to action_steps
        count = 0
        for step in step_text_list:
            action_step = self.parse_one_action_per_step(step, count)
            action_step_list = action_step_list + [action_step]
            count +=1

        #now order them, going through the step list
        all_steps = " , ".join(step_text_list)

        #action_step_list = self.order_ids_stepwise(action_step_list)

        return action_step_list

    #creates dictionary, where key = entity, value = list of actions entity refers to
    def resolve_refs_after_parse(self, step_text_list, action_step_list):

        #next do coreference resolution
        sentence = ", then ".join(step_text_list)
        
        #parses and finds coreferences
        doc = self.nlp(sentence)

        #go over each word
        for token in doc:

            if token._.in_coref:
                for cluster in token._.coref_clusters:
                    #the thing it refers to
                    main = cluster.main
                
                    #main the token this one refers to

                    #find action step token is in, 
                    #find the action step main is in, and set the reference id
                    token_action_id = self.find_action_step_for_token(token, action_step_list)
                    ref_action_id = self.find_action_step_for_token(main, action_step_list)

                    #swap so we refer to a previous step
                    if token_action_id < ref_action_id:
                        tmp = token_action_id
                        token_action_id = ref_action_id
                        ref_action_id = tmp

                    #looks at entities corresponding to this token for this action step, and inserts the
                    #action ID they refer to for them
                    if token_action_id != -1 and token_action_id != ref_action_id:
                        action_step_list[token_action_id].set_references(token, ref_action_id)

                    #ref_dict[token].append(main)
        return action_step_list

    #main run method
    #pass a list of steps (strings)
    #this will parse them all, and resolve references
    #returning action_step_list, of ActionStep objects
    def parse_and_resolve_all_refs(self, step_list):
        #parse the steps
        action_step_list = rr_model.parse_step_list(step_list)
        
        #now do reference resolution
        action_step_list = rr_model.resolve_refs_after_parse(step_list, action_step_list)
        
        return action_step_list
    
    def print_action_step_list(self, step_list, action_step_list):
        count = 0
        for action_step in action_step_list:
            print("\nStep Annotation: {}".format(step_list[count]))
            print("Action ID: {}".format(action_step.act_id))
            print("PRED: {}".format(action_step.pred))

            dobj_list_str = [(dobj.ent_text.text + " (Refers to action ID: {})".format(dobj.act_id_ref)) for dobj in action_step.dobj_list]
            pp_list_str = [(pp.ent_text.text + " (Refers to action ID: {})".format(pp.act_id_ref)) for pp in action_step.pp_list]

            print("DOBJ: {}".format(dobj_list_str))
            print("PP: {}".format(pp_list_str))

            count+=1
        return

rr_model = ReferenceResolver()

eg_text_1 = "Grab the spatula and use it to mix it with the dried onions." 
eg_text_2 = "drizzle little bit of olive oil on both the sides of 4 bread slices"

#act_ent_dict =rr_model.parse_action_entity(eg_text_2)

#print(act_ent_dict.keys())
#print(act_ent_dict.values())

#NOTE: action step pred is a string, but for Entities, the ent_text is a Spacy object. use .text to get actual
#string

"""
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

"""


step_text_eg = ['pour into the ingredients',
 'pour into the ingredients',
 'stir the mixture',
 'pour in after mixing it',
 'decorate with fruit']

step_text_eg_2 = ['Preheat the oven at 425 degree', 
'drizzle little bit of olive oil on both the sides of 4 bread slices',
'Lay the bacon slices on a broiler pan',
'place both bread and bacon in the oven',
'cook for 10-15 minutes',
'cut the avocado into half',
'take off the seed',
'scoop the pulp of half into a bowl',
'Squeeze half of a lemon juice',
'add a pinch of salt and some fresh parsley',
'mash them all together with a fork to get a paste',
'Slice the other half of the avocado',
' take out the bacon and bread toast from the oven',
'To assemble bottom layer of sandwich',
'take 2 slices of bread',
'spread the avocado mixture over',
'Place some tomato slices on the avocado spread and season with little salt and pepper',
'season with little salt and pepper',
'Drizzle a little bit of olive oil over it',
'Top it with the bacon slices',
'on the other slices of bread, place the sliced avocadoes',
'Season avocado slices with little bit of salt',
'top the avocadoes with lettuce leaves',
'Place the avocado and lettuce topped bread slice on the bacon and tomato topped bread slice',
'serve cutting the sandwich into 2 pieces']

#print(step_text_eg_2)

action_step_list = rr_model.parse_and_resolve_all_refs(step_text_eg_2)

#assumes 1 action per step!
rr_model.print_action_step_list(step_text_eg_2, action_step_list)

#rr_model.write_action_step_list_file(action_step_list, './examples/action_step_list')