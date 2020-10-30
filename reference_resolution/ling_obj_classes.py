import spacy


#defines an action step, which has 
# a single predicate (text), a list of direct objects for that predicate (list of Entities)
# and a list of propositional phrases for that predicate (list of Entities)
class ActionStep:
    
    def __init__(self, act_id = None, pred = None, dobj_list = None, pp_list = None):
        if act_id is None:
            act_id = -1    
        self.act_id = act_id

        self.pred = pred
        
        if dobj_list is None:
            dobj_list = []
        self.dobj_list = dobj_list
        
        if pp_list is None:
            pp_list = []    
        self.pp_list = pp_list

        return

    def addPP(self, pp):
        self.pp_list.append(pp)
        return

    def addDOBJ(self, dobj):
        self.dobj_list.append(dobj)
        return
    
    #use this to re-order action steps
    def setActId(self, act_id):
        self.act_id = act_id
        return    

    #returns bool if in DOBJ List
    def tokenInDOBJList(self, token):
        for dobj in self.dobj_list:
            if token.text in dobj.ent_text.text:
                return True
        
        return False
    
    #returns bool if in DOBJ List
    def tokenInPPList(self, token):
        for pp in self.pp_list:
            if token.text in pp.ent_text.text:
                return True
        
        return False
    
    def hasTokenEntity(self, token):
        return (self.tokenInDOBJList(token) or self.tokenInPPList(token))

    #looks through PP and DOBJ list, and if the token is in them, 
    #sets the corresponding entities reference ID to passed ID
    def set_references(self, token, ref_id):
        ent_list = self.pp_list + self.dobj_list

        for ent in ent_list:
            #if an entity contains the token
            if token.text in ent.ent_text.text:
                ent.set_reference(ref_id)
        return 

    #to check if a object is in the DOBJ or PP list of ActionStep
    def __contains__(self, obj):
        in_dobj = any([obj.text in dobj.ent_text.text for dobj in self.dobj_list])
        in_pp = any([obj.text in pp.ent_text.text for pp in self.pp_list])

        if in_dobj or in_pp:
            return True
        else:
            return False

    def __str__(self):
        s = ''
        s += 'Action ID: {}\n'.format(self.act_id)
        s += 'Predicate: {}\n'.format(self.pred)

        s += 'Direct Objects:\n'
        #print(len(self.dobj_list))
        for dobj in self.dobj_list:
            s += str(dobj) + '\n'

        s += 'Propositional Phrases:\n'
        for pp in self.pp_list:
            s += str(pp) + '\n'

        return s

#entity class, has the text for the entity, the entity type (DOBJ or PP)
# and the action id it reference 
class Entity:
    def __init__(self, ent_text, ent_type, act_id_ref = None):
        self.ent_text = ent_text
        self.ent_type = ent_type

        if act_id_ref is None:
            act_id_ref = -1
        self.act_id_ref = act_id_ref
        
        return
    #given an action step, sets the reference of this entity to that action step
    def set_reference(self, ref_act_id):
        self.act_id_ref = ref_act_id
        return

    def __str__(self):
        return 'Type: {}, Entity Text: {}, Action ID: {}'.format(self.ent_type, self.ent_text, self.act_id_ref)

    