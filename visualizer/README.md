## Action Graph Visualizer

### Purpose
Visualize JSON action graphs using Graphviz. Requires Action Graph in JSON format and all associated frames to be saved on disk.


### Usage
#### `python`
```
# ensure action graph in .json format is available, along with any image files used
# generated graph will be in visualizer-output/graph.pdf

from visualizer import file_to_viz
file_to_viz('<path_to_action_graph>.json')
```

#### `CLI`
```
python visualizer.py <path_to_action_graph>.json
```


#### Action Graph Format
Action graph should be a JSON file with a list of Action Steps of the following format:
```
{
    "annot": annotation (string),
    "img": image_path (string of absolute path or relative path to pwd),
    "pred": predicate (string),
    "entities": entity_list (list of strings),
    "bboxes": bounding_box_list (list of coordinates {left (int), bottom (int), right (int), top (int)}),
    "ea": entity_action_step_edge (list of action_id (int) corresponding to each element in entity_list, 0 indexed, -1 for no edge),
    "eb": entity_bounding_box_edge (list of indices (int) into bounding_box_list corresponding to each element in entity_list, 0 indexed, -1 for no edge)
}
```


