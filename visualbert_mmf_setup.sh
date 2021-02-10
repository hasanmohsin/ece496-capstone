#!/bin/bash

#should start from repo directory root

mkdir model_data
wget -O /content/model_data/answers_vqa.txt https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt
wget -O /content/model_data/vocabulary_100k.txt https://dl.fbaipublicfiles.com/pythia/data/vocabulary_100k.txt
wget -O /content/model_data/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth 
wget -O /content/model_data/pythia.pth https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.pth
wget -O /content/model_data/pythia.yaml https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.yml
wget -O /content/model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
wget -O /content/model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf /content/model_data/detectron_weights.tar.gz

pip install yacs cython matplotlib
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

git clone https://github.com/facebookresearch/mmf.git mmf
cd ./mmf
# Don't modify torch version
sed -i '/torch/d' requirements.txt
pip install -e .

PATH=$PATH:~/./mmf

# Install maskrcnn-benchmark to extract detectron features
git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git
cd /./vqa-maskrcnn-benchmark
# Compile custom layers and build mask-rcnn backbone
python setup.py build
python setup.py develop
PATH=$PATH:~/./vqa-maskrcnn-benchmark



