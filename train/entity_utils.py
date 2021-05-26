import torch

#HELPER FUNCTIONS FOR REPRESENTING ENTITY AS AVERAGE OF ITS FEATURES 
def contains_(sub, pri):
    M, N = len(pri), len(sub)
    i, LAST = 0, M-N+1
    while True:
        try:
            found = pri.index(sub[0], i, LAST) # find first elem in sub
        except ValueError:
            return False
        if pri[found:found+N] == sub:
            return [found, found+N-1]
        else:
            i = found+1

#given steps_list (Batch x string)
#entity_list (Batch x num_actions x num_entities)
#returns a (Batch x num_entities in video x 2) list 
#where ent_inds[b][i] = [entity_start ind, entity_end ind] (inclusive) inside overall
#tokenized text ids/embeddings
#NOTE: assumes steps_list has no [unused3] id 
def get_ent_inds(model, entity_list, steps_list):
    
    #print(entity_list)
    
    entity_overall_index_list = []
    
    #go over each batch
    for b in range(len(steps_list)):
        
        entity_batch_index_list = []
        
        vid_step_list = steps_list[b]        
        action_list = vid_step_list.split('. ')
                        
        #the tokenized input for step list (without [unused3] token) 
        inputs_overall = model.lxmert_tokenizer(
            vid_step_list,
            padding="longest",
            truncation=False,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        #includes [CLS], and [SEP]
        text_tokens_overall = model.lxmert_tokenizer.convert_ids_to_tokens(inputs_overall.input_ids[0])
        
        #print(text_tokens_overall)
        
        count = 0
        
        #go over each action 
        for action in action_list:
            action_inputs = model.lxmert_tokenizer(
                action,
                padding="longest",
                truncation=False,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

            #input_ids of size 1 x num_tokens in action step
            action_tokens_step = model.lxmert_tokenizer.convert_ids_to_tokens(action_inputs.input_ids[0])[1:-1]
            
            #print(action_tokens_step)
            
            #entities for that 
            entities_in_action = entity_list[b][count]
            for e in entities_in_action:
                #tokenize the entity
                entity_input = model.lxmert_tokenizer(
                    e,
                    padding="longest",
                    truncation=False,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )

                e_text_tokens =  model.lxmert_tokenizer.convert_ids_to_tokens(entity_input.input_ids[0])[1:-1]
                
                #print(e)
                #print(e_text_tokens)

                #first check where action ids occur in overall step
                action_ind_start, action_ind_end = contains_(action_tokens_step, text_tokens_overall)
                ent_ind_start, ent_ind_end = contains_(e_text_tokens, text_tokens_overall[action_ind_start:action_ind_end+1])
                ent_ind_start = ent_ind_start+ action_ind_start
                ent_ind_end = ent_ind_end+ action_ind_start
                
                #print(action_ind_start, action_ind_end)
                #print(ent_ind_start, ent_ind_end)
                
                #strip away inds which don't occur inside action step

                ent_ind = [ent_ind_start, ent_ind_end]
                entity_batch_index_list.append(ent_ind)
            
            count+=1    
        entity_overall_index_list.append(entity_batch_index_list)

    
    return entity_overall_index_list


    

def get_entity_embeddings(language_output, entity_ind_list):
    batch_size = language_output.size()[0]
    
    #in batch
    num_entities_batch = [len(b_ind_list) for b_ind_list in entity_ind_list]
    num_entities = sum(num_entities_batch)
        
    entity_embeddings = torch.Tensor(num_entities, language_output.size(-1)).cuda()
    count = 0
    
    for b in range(batch_size):
        b_output = language_output[b, :]
        b_ind_list = entity_ind_list[b]
       
        for e_inds in b_ind_list:
            entity_embed = b_output[e_inds[0]:e_inds[1]+1,:].mean(0).squeeze()
            entity_embeddings[count,:] = entity_embed
            count +=1
    
    return entity_embeddings


# def get_ent_inds(model, entity_list, steps_list):
    
#     print(entity_list)
    
#     entity_overall_index_list = []
    
#     #go over each batch
#     for b in range(len(steps_list)):
        
#         entity_batch_index_list = []
        
#         vid_step_list = steps_list[b]        
#         action_list = vid_step_list.split('[unused3]')[:-1]
        
#         print(action_list)
        
#         #the tokenized input for step list (without [unused3] token) 
#         inputs_overall = model.lxmert_tokenizer(
#             vid_step_list,
#             padding="longest",
#             truncation=False,
#             return_token_type_ids=True,
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors="pt"
#         )
        
#         #includes [CLS], and [SEP]
#         text_tokens_overall = model.lxmert_tokenizer.convert_ids_to_tokens(inputs_overall.input_ids[0])
        
#         #print(text_tokens_overall)
        
#         count = 0
        
#         #go over each action 
#         for action in action_list[:-1]:
#             action_inputs = model.lxmert_tokenizer(
#                 action,
#                 padding="longest",
#                 truncation=False,
#                 return_token_type_ids=True,
#                 return_attention_mask=True,
#                 add_special_tokens=True,
#                 return_tensors="pt"
#             )

#             #input_ids of size 1 x num_tokens in action step
#             action_tokens_step = model.lxmert_tokenizer.convert_ids_to_tokens(action_inputs.input_ids[0])[1:-1]
            
#             #print(action_tokens_step)
            
#             #entities for that 
#             print(b, count)
#             entities_in_action = entity_list[b][count]
#             for e in entities_in_action:
#                 #tokenize the entity
#                 entity_input = model.lxmert_tokenizer(
#                     e,
#                     padding="longest",
#                     truncation=False,
#                     return_token_type_ids=True,
#                     return_attention_mask=True,
#                     add_special_tokens=True,
#                     return_tensors="pt"
#                 )

#                 e_text_tokens =  model.lxmert_tokenizer.convert_ids_to_tokens(entity_input.input_ids[0])[1:-1]
                
#                 #print(e)
#                 #print(e_text_tokens)

#                 #first check where action ids occur in overall step
#                 action_ind_start, action_ind_end = contains_(action_tokens_step, text_tokens_overall)
#                 ent_ind_start, ent_ind_end = contains_(e_text_tokens, text_tokens_overall[action_ind_start:action_ind_end+1])
#                 ent_ind_start = ent_ind_start+ action_ind_start
#                 ent_ind_end = ent_ind_end+ action_ind_start
                
#                 #print(action_ind_start, action_ind_end)
#                 #print(ent_ind_start, ent_ind_end)
                
#                 #strip away inds which don't occur inside action step

#                 ent_ind = [ent_ind_start, ent_ind_end]
#                 entity_batch_index_list.append(ent_ind)
            
#             count+=1    
#         entity_overall_index_list.append(entity_batch_index_list)

    
#     return entity_overall_index_list