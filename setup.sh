#!/bin/bash

# Make virtual environment.
python3 -m venv env
source env/bin/activate

# Install all needed modules from pip.
# https://stackoverflow.com/questions/22250483/stop-pip-from-failing-on-single-package-when-installing-with-requirements-txt
cat requirements.txt | xargs -n 1 pip install

# Install remaining modules from source.
./scripts/neuralcoref.sh
./scripts/detectron.sh
./scripts/mmf.sh
./scripts/maskrcnn.sh
