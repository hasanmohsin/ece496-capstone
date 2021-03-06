# Source: https://colab.research.google.com/drive/18TyuMfZYlgQ_nXo-tr8LCnzUaoX0KS-h?usp=sharing#scrollTo=ROYK62wS82ug.
# Some modifications have been made.

import PIL.Image
import detector_utils
import io
import numpy as np
import torch

from IPython.display import clear_output, Image, display
from detector_utils.processing_image import Preprocess
from detector_utils.visualizing_image import SingleImageViz
from detector_utils.modeling_frcnn import GeneralizedRCNN
from detector_utils.utils import Config

class Detector():
    def __init__(self, device):
        self.device = device

        config = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        if torch.cuda.is_available():
            config.model.device = "cuda"
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=config)

        self.image_preprocess = Preprocess(config)
        
    def inference(self, URL, max_detections=1, visualize=False):
        images, sizes, scales_yx = self.image_preprocess(URL)

        output = self.frcnn(
            images, 
            sizes, 
            scales_yx=scales_yx, 
            padding="max_detections",
            max_detections=max_detections,
            return_tensors="pt",
            location=self.device
        )

        normalized_boxes = output.get("normalized_boxes")
        features = output.get("roi_features")
        
        if visualize:
            self.visualize(URL, output)
        
        return normalized_boxes, features
        
    def visualize(self, URL, output):
        visualizer = SingleImageViz(URL, id2obj=None, id2attr=None)
        visualizer.draw_boxes (
            output.get("boxes"),
            output.pop("obj_ids"),
            output.pop("obj_probs"),
            output.pop("attr_ids"),
            output.pop("attr_probs"),
        )
        
        self.showarray(visualizer._get_buffer())

    def showarray(self, a, fmt='jpeg'):
        a = np.uint8(np.clip(a, 0, 255))
        f = io.BytesIO()
        PIL.Image.fromarray(a).save(f, fmt)
        display(Image(data=f.getvalue()))
