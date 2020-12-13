# ece496-capstone: Videos to Action Graphs

Project for ECE496 capstone, University of Toronto.

The project is about building a model which takes as input a video, with a time-aligned transcript, and outputs a visually grounded action graph (like a scene graph, but higher level, and with temporal relations). 

The current approach is to parse the transcript for entities and actions, then use reference resolution to find temporal relations between steps. These entities are also visually grounded (since visually grounding the text can help improve the model's competence at finding temporal relations.


**Google Docs Links**:

Resources Doc:
https://docs.google.com/document/d/1GmozMEyUohby9WvtRHeRLLtqIByjGyLvtbVmZpY4UYQ/edit?usp=sharing

