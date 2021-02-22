import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd


import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO
from argparse import Namespace


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict


from mmf.datasets.processors.processors import VocabProcessor, VQAAnswerProcessor
from mmf.models.pythia import Pythia
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.env import setup_imports
from mmf.utils.configuration import Configuration

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

setup_imports()

class VL_Model:
  TARGET_IMAGE_SIZE = [448, 448]
  CHANNEL_MEAN = [0.485, 0.456, 0.406]
  CHANNEL_STD = [0.229, 0.224, 0.225]
  
  def __init__(self):
    self._init_processors()
    self.visual_bert = registry.get_model_class(
            "visual_bert"
        ).from_pretrained(
            "visual_bert.pretrained.coco"
        )

    # Add this option so that it only output hidden states
    self.visual_bert.model.output_hidden_states = True

    self.visual_bert.model.to("cuda")
    self.visual_bert.model.eval()

    # Add this option so that losses are not pushed into output
    self.visual_bert.training_head_type = "finetuning"

    self.detection_model = self._build_detection_model()
    
  def _init_processors(self):
    args = Namespace()
    args.opts = [
        "config=projects/pythia/configs/vqa2/defaults.yaml",
        "datasets=vqa2",
        "model=visual_bert",
        "evaluation.predict=True"
    ]
    args.config_override = None

    configuration = Configuration(args=args)
    
    config = self.config = configuration.config
    vqa2_config = config.dataset_config.vqa2
    text_processor_config = vqa2_config.processors.text_processor
    
    text_processor_config.params.vocab.vocab_file = "../model_data/vocabulary_100k.txt"

    # Add preprocessor as that will needed when we are getting questions from user
    self.text_processor = VocabProcessor(text_processor_config.params)

    registry.register("coco_text_processor", self.text_processor)
  

  def _multi_gpu_state_to_single(self, state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            raise TypeError("Not a multiple GPU state of dict")
        k1 = k[7:]
        new_sd[k1] = v
    return new_sd
  
  def predict(self, url, text):
    with torch.no_grad():
      detectron_features = self.get_detectron_features(url)

      sample = Sample()

      processed_text = self.text_processor({"text": text})
      #sample.text = processed_text["text"]
      sample.text_len = len(processed_text["tokens"])

      encoded_input = tokenizer(text, return_tensors='pt')
      sample.input_ids = encoded_input.input_ids
      sample.input_mask = encoded_input.attention_mask
      sample.segment_ids = encoded_input.token_type_ids

      sample.image_feature_0 = detectron_features
      sample.image_info_0 = Sample({
          "max_features": torch.tensor(100, dtype=torch.long)
      })

      sample_list = SampleList([sample])
      sample_list = sample_list.to("cuda")

      output = self.visual_bert(sample_list)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return output
    
  
  def _build_detection_model(self):

      cfg.merge_from_file('../model_data/detectron_model.yaml')
      cfg.freeze()

      model = build_detection_model(cfg)
      checkpoint = torch.load('../model_data/detectron_model.pth', 
                              map_location=torch.device("cpu"))

      load_state_dict(model, checkpoint.pop("model"))

      model.to("cuda")
      model.eval()
      return model
  
  def get_actual_image(self, image_path):
      if image_path.startswith('http'):
          path = requests.get(image_path, stream=True).raw
      else:
          path = image_path
      
      return path

  def _image_transform(self, image_path):
      path = self.get_actual_image(image_path)

      img = Image.open(path)
      im = np.array(img).astype(np.float32)
      im = im[:, :, ::-1]
      im -= np.array([102.9801, 115.9465, 122.7717])
      im_shape = im.shape
      im_size_min = np.min(im_shape[0:2])
      im_size_max = np.max(im_shape[0:2])
      im_scale = float(800) / float(im_size_min)
      # Prevent the biggest axis from being more than max_size
      if np.round(im_scale * im_size_max) > 1333:
           im_scale = float(1333) / float(im_size_max)
      im = cv2.resize(
           im,
           None,
           None,
           fx=im_scale,
           fy=im_scale,
           interpolation=cv2.INTER_LINEAR
       )
      img = torch.from_numpy(im).permute(2, 0, 1)
      return img, im_scale


  def _process_feature_extraction(self, output,
                                 im_scales,
                                 feat_name='fc6',
                                 conf_thresh=0.2):
      batch_size = len(output[0]["proposals"])
      n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
      score_list = output[0]["scores"].split(n_boxes_per_image)
      score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
      feats = output[0][feat_name].split(n_boxes_per_image)
      cur_device = score_list[0].device

      feat_list = []

      for i in range(batch_size):
          dets = output[0]["proposals"][i].bbox / im_scales[i]
          scores = score_list[i]

          max_conf = torch.zeros((scores.shape[0])).to(cur_device)

          for cls_ind in range(1, scores.shape[1]):
              cls_scores = scores[:, cls_ind]
              keep = nms(dets, cls_scores, 0.5)
              max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                           cls_scores[keep],
                                           max_conf[keep])

          keep_boxes = torch.argsort(max_conf, descending=True)[:100]
          feat_list.append(feats[i][keep_boxes])
      return feat_list

  def masked_unk_softmax(self, x, dim, mask_idx):
      x1 = F.softmax(x, dim=dim)
      x1[:, mask_idx] = 0
      x1_sum = torch.sum(x1, dim=1, keepdim=True)
      y = x1 / x1_sum
      return y
   
    
  def get_detectron_features(self, image_path):
      im, im_scale = self._image_transform(image_path)
      img_tensor, im_scales = [im], [im_scale]
      current_img_list = to_image_list(img_tensor, size_divisible=32)
      current_img_list = current_img_list.to('cuda')
      with torch.no_grad():
          output = self.detection_model(current_img_list)
      feat_list = self._process_feature_extraction(output, im_scales, 
                                                  'fc6', 0.2)
      return feat_list[0]

  def get_detectron_features_and_out(self, image_path):
      im, im_scale = self._image_transform(image_path)
      img_tensor, im_scales = [im], [im_scale]
      current_img_list = to_image_list(img_tensor, size_divisible=32)
      current_img_list = current_img_list.to('cuda')
      with torch.no_grad():
          output = self.detection_model(current_img_list)
      feat_list = self._process_feature_extraction(output, im_scales, 
                                                  'fc6', 0.2)
      return feat_list[0], output[0]["proposals"], im
    

  def get_detectron_features_and_out(self, image_path):
      im, im_scale = self._image_transform(image_path)
      img_tensor, im_scales = [im], [im_scale]
      current_img_list = to_image_list(img_tensor, size_divisible=32)
      current_img_list = current_img_list.to('cuda')
      with torch.no_grad():
          output = self.detection_model(current_img_list)
      feat_list = self._process_feature_extraction(output, im_scales, 
                                                  'fc6', 0.2)
      return feat_list[0], output[0]["proposals"], im
  
  
  def _process_bbox_extraction(self, output,
                                 im_scales,
                                 conf_thresh=0.2):
      batch_size = len(output[0]["proposals"])
      n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
      score_list = output[0]["scores"].split(n_boxes_per_image)
      score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
      cur_device = score_list[0].device

      bbox_list = []

      for i in range(batch_size):
          bboxes = output[0]["proposals"][i].bbox
          dets = output[0]["proposals"][i].bbox / im_scales[i]
          scores = score_list[i]

          max_conf = torch.zeros((scores.shape[0])).to(cur_device)

          for cls_ind in range(1, scores.shape[1]):
              cls_scores = scores[:, cls_ind]
              keep = nms(dets, cls_scores, 0.5)
              max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                           cls_scores[keep],
                                           max_conf[keep])

          keep_boxes = torch.argsort(max_conf, descending=True)[:100]
          bbox_list.append(bboxes[keep_boxes])
      return bbox_list

  def detectron_get_bbox(self, image_path):
        im, im_scale = self._image_transform(image_path)
        img_tensor, im_scales = [im], [im_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to('cuda')
        with torch.no_grad():
            output = self.detection_model(current_img_list)
        bbox_list = self._process_bbox_extraction(output, im_scales, 0.2)
        return bbox_list[0]

    #returns list of bounding box coordinates, in order of input entity_list
    #bb_embed_list assumed to be 100 by 768 (the embedding size)
  def visual_ground(self, output, entity_list, bbox_list, full_sentence, step_sentence):
        
        bbox_for_entity = []

        encoded_input = tokenizer(full_sentence, return_tensors='pt')
        step_encoded_input = tokenizer(step_sentence, return_tensors='pt')

        all_tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
        step_tokens = tokenizer.convert_ids_to_tokens(step_encoded_input["input_ids"][0])

        #find start of step_token list in all_tokens
        start_step_ind = len(all_tokens) - len(step_tokens) +1
        step_tokens = all_tokens[start_step_ind:]

        bb_embed_list = output['sequence_output'][0, len(encoded_input["input_ids"][0]):].cpu().detach().numpy()
        text_embed_list = output['sequence_output'][0, 0:len(encoded_input["input_ids"][0])].cpu().detach().numpy()

        #move through the entities, we will form a meta-embeding for each entity
        # by averaging the embeddings given by BERT (since an entity could be a longer string of many sub-word/tokens)
        for entity in entity_list:
            #print("\n{}".format(entity))
            #run through sub-words that are in the step, if its a part of the entity, add it
            related_embeddings=  [text_embed_list[start_step_ind + i] for i in range(len(step_tokens)) if check_subword_in_word(step_tokens[i], entity)]
            
            #print(len(related_embeddings))
            related_embeddings = np.array(related_embeddings)
            #print(len(related_embeddings))
            #print(np.array(related_embeddings).shape)

            
            #average over them to get embedding for entity
            entity_embedding = np.mean(related_embeddings, axis = 0)

            #print(entity_embedding)

            #now we have the entity embedding, compare to all bounding boxes to get scores for alignment
            #print(bb_embed_list.shape)
            #print(entity_embedding.shape)
            scores = bb_embed_list@entity_embedding
            bbox_ind = np.argmax(scores)
            #scores = scores/np.sum(scores)

            bbox_for_entity.append(bbox_list[bbox_ind].cpu().detach().numpy())
        
        return bbox_for_entity


#pass subword (from BERT tokenizer) string, and word string
#true if subword came from word
def check_subword_in_word(subword, string):
    #entity may be string of multiple words
    words = string.split()

    if (subword[:2] == "##" and any([subword[2:] in word for word in words])) or (subword[-2:] == "##" and any([subword[-2:] in word for word in words])) or (subword in words):
        return True
    else:
        return False

def get_im_resize_factor(img_path): 
    img = Image.open(img_path)
    
    im = np.array(img).astype(np.float32)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)
    #im = cv2.resize(
    #    im,
    #    None,
    #    None,
    #    fx=im_scale,
    #    fy=im_scale,
    #    interpolation=cv2.INTER_LINEAR
    #)

    return im_scale


#takes in video, and sets bb values for entities
def vg_inference(video):
    demo = VL_Model()

    step_text = [step.text for step in video.steps]

    #assuming all of same size
    im_scale = get_im_resize_factor(video.steps[0].path)

    for step_num in range(len(video.steps)):
        full_sentence = " ".join(step_text[:step_num+1])
        sentence = video.steps[step_num].text

        #construct entity list for step
        entity_list = []

        #add dobjs
        entity_list.extend([video.steps[step_num].DOBJ[i].text for i in range(len(video.steps[step_num].DOBJ))])
        #add pps
        entity_list.extend([video.steps[step_num].PP[i].text for i in range(len(video.steps[step_num].PP))])


        img_path = video.steps[step_num].path
        output = demo.predict(img_path, full_sentence)

        bbox_list = demo.detectron_get_bbox(img_path)

        #list of bbox coordinates for scaled image
        entity_bb = demo.visual_ground(output = output, 
                                entity_list = entity_list, 
                                bbox_list = bbox_list, 
                                full_sentence = full_sentence, 
                                step_sentence = sentence)
        
        for bb_ind in range(len(entity_bb)):
            entity_bb[bb_ind] = np.round(entity_bb[bb_ind]/im_scale).astype(int)


        #set bbox coordinates for all entities
        count = 0
        #re-order and put in dict for step
        for dobj in video.steps[step_num].DOBJ:
            dobj.bb = {'left': int(entity_bb[count][0]), 'top': int(entity_bb[count][3]), 'bot': int(entity_bb[count][1]), 'right': int(entity_bb[count][2])}
            count += 1

        for pp in video.steps[step_num].PP:
            pp.bb = {'left': int(entity_bb[count][0]), 'top': int(entity_bb[count][3]), 'bot': int(entity_bb[count][1]), 'right': int(entity_bb[count][2])}
            count+=1

    return


### Inference Routine:
# video = Video(video_path, transcript_path)
#  video.align()

#  rr = RR(video.steps)
#  rr.run()

  # Generate frames.
#  video.generate_frames('graph')

  #set bboxes for video entities
#  vg_inference(video)

#  return video