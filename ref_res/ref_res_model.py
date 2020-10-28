import spacy
import neuralcoref

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
        sentence = ", then ".join(step_txt)
        
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

                        #we stop at first verb up
                        while action.pos_ != "VERB":
                            action = action.head

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
                
rr_model = ReferenceResolver()

act_ent_dict =rr_model.parse_action_entity("Grab the spatula and use it to mix it with the dried onions.")

print(act_ent_dict.keys())
print(act_ent_dict.values())