#!/bin/bash

# Install maskrcnn benchmark.

git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git
cd ./vqa-maskrcnn-benchmark
python setup.py build
python setup.py develop

cd ..
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
