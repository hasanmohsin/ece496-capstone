import webvtt
import json

from torchvision.io import read_video

def get_frame(start, end, vframes):
  middle = datetime.today() + start + (end - start)
  middle = int(middle.hour * 3600 + middle.minute * 60 + middle.second)
  vframe = vframes[middle]
  return vframe

class Video:

  def __init__(self, video_path, transcript_path):
    # Get all of information from the video.
    self.vframes, self.aframes, self.info = read_video(video_path, pts_unit='sec')

    # Get the FPS. Note that sometimes the file may not contain metadata causing
    # this to fail. Ensure that the video contains metadata!
    self.fps = int(self.info.get('video_fps'))

    if not self.fps:
      raise Exception('Video {} does not contain required metadata.'.format(video_path))

    # Change the axes from [T, H, W, C] -> [T, C, H, W].
    self.vframes = self.vframes.permute(0, 3, 1, 2)

    # Parse through the transcript.
    self.captions = webvtt.read(transcript_path)
    
    # We haven't aligned the frames yet.
    self.vframes_aligned = None

  def downsample(self):
    # Downsample by striding along the array.
    self.vframes = self.vframes[::self.fps]

  def align(self):
    self.downsample()
    self.steps = [Step(idx, caption, self.vframes) for idx, caption in enumerate(self.captions)]
    self.vframes_aligned = [step.vframe for step in self.steps]

  def generate_frames(self, path, swap=False):
    for step in self.steps:
      step.generate_frames(path, swap)

  def generate_json(self, file):
    info = [step.generate_json() for step in self.steps]
    json.dump(info, open('{}.json'.format(file), 'w'))


# --------------------------------------------


from datetime import datetime
from torchvision.utils import save_image

import os
import numpy as np

class Step:

  time_format = '%H:%M:%S.%f'
  print_format = '%H:%M:%S'
  offset = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

  def __init__(self, idx, caption, vframes):
    self.idx = idx
    self.text = caption.text
    self.start = datetime.strptime(caption.start, Step.time_format)
    self.end = datetime.strptime(caption.end, Step.time_format)

    self.set_frames(vframes)
    self.set_frame()

    # Set by the model.
    self.DOBJ = []
    self.PRED = None
    self.PP = []

    self.path = None

  def generate_frames(self, path, swap):
    self.path = os.path.abspath('./{}/{}.png'.format(path, self.idx))

    if not swap:
        save_image(self.vframe.float() / 255, self.path)
    else:
        temp = self.vframe[[2, 1, 0], :, :]
        save_image(temp.float() / 255, self.path)

  @staticmethod
  def get_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second

  def get_interval(self):
    start_index = self.get_seconds(self.start)
    end_index = self.get_seconds(self.start + (self.end - self.start))
    return (start_index - 1), (end_index - 1)
  
  def set_frames(self, vframes):
    start_index, end_index = self.get_interval()
    self.vframes = vframes[start_index:(end_index + 1)]

  def get_index(self):
    index = int(len(self.vframes) / 2)
    return index

  def set_frame(self):
    self.vframe = self.vframes[self.get_index()]

  def __str__(self):
    s = 'Action ID {}: {} ({} -> {})\n'.format(self.idx, self.text, self.start.strftime(Step.print_format), self.end.strftime(Step.print_format))
    s += 'Predicate: {}\n'.format(self.PRED)

    for DOBJ in self.DOBJ:
      s += 'DOBJ: {} ({}), BB: {}\n'.format(DOBJ.text, DOBJ.reference, DOBJ.bb)

    for PP in self.PP:
      s += 'PP: {} ({}), BB: {}\n'.format(PP.text, PP.reference, PP.bb)

    return s

  def generate_json(self):
    attr = dict()
    attr['annot'] = self.text
    attr['img'] = self.path
    attr['pred'] = self.PRED

    attr['entities'] = []
    attr['bboxes'] = []
    attr['ea'] = []
    attr['eb'] = []

    for idx, entity in enumerate(self.DOBJ + self.PP):
      attr['entities'].append(entity.text)
      attr['bboxes'].append(entity.bb)
      attr['ea'].append(entity.reference)
      attr['eb'].append(-1 if not entity.bb else idx)

    return attr


# --------------------------------------------


class Object:

  def __init__(self, step, text, bb=None, reference=-1):
    self.step = step
    self.text = text
    self.reference = reference
    self.bb = bb
