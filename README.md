# ece496-capstone: Videos to Action Graphs

Videos to Action Graphs - Project for ECE496 capstone, University of Toronto.

The project is about building a model which takes as input a video, with a time-aligned transcript, and outputs a visually grounded action graph (like a scene graph, but higher level, and with temporal relations). 

The current approach is to parse the transcript for entities and actions, then use reference resolution to find temporal relations between steps. These entities are also visually grounded (since visually grounding the text can help improve the model's competence at finding temporal relations).

## Setup

To get started, run `setup.sh` from the project directory. This will create a virtual Python environment `venv`, and install all of the necessary modules. In order to use the modules installed from source, run `source scripts/env.sh` to append the directory paths to the system path. Alternatively, the directory paths can be added to your default shell configuration file. 

When you run the model inference for the first time, you will likely encounter `PermissionError` which is due to a problem with the configuration files. Run `scripts/config.sh` to fix this. The `mmf` module uses the following environment variables: `MMF_CACHE_DIR`, `MMF_DATA_DIR`, `MMF_SAVE_DIR`. If you set these environment variables, you may run into other errors.


**Google Docs Links**:

Resources Doc:
https://docs.google.com/document/d/1GmozMEyUohby9WvtRHeRLLtqIByjGyLvtbVmZpY4UYQ/edit?usp=sharing

