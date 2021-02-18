#!/bin/bash

# Install neuralcoref.

git clone https://github.com/huggingface/neuralcoref.git

pip install -U spacy
python -m spacy download en

cd neuralcoref
pip install -r requirements.txt
pip install -e .

# This download is needed to match the proper version.

python -m spacy download en_core_web_sm
