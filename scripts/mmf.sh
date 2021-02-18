#!/bin/bash

# Install mmf module.

git clone https://github.com/facebookresearch/mmf.git
cd mmf
sed -i '/torch/d' requirements.txt
pip install --editable .
