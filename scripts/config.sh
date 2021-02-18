#!/bin/bash

# This script overrides the normal configuration file for VisualBERT. It is needed
# because the default configuration file contains incorrect absolute paths.

script_dir=$(dirname $0)

cp $script_dir/../configs/visual_bert.pretrained.coco.defaults/config.yaml ~/.cache/torch//mmf/data/models/visual_bert.pretrained.coco.defaults
