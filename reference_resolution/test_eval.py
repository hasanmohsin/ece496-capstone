import spacy
import neuralcoref

from ling_obj_classes import *
from ref_res_model import *
from evaluation import *


rr_model = ReferenceResolver()

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

evaluate(action_step_list, action_step_list)