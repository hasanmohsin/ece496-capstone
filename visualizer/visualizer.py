from graphviz import Graph, Digraph, nohtml
import imagesize
import json
import sys


# entry point
def file_to_viz(path='output.json', debug=False):
    json_to_viz(read_json(path), debug)


# checks for valid JSON format and reads content
def read_json(path='output.json'):
    file = open(path)
    line = file.read().replace('\n', ' ')
    file.close()
    try:
        parsed_json = json.loads(line)
    except:
        assert False, 'Invalid JSON'
    return parsed_json


# wrap text lines after 32 characters to fit html components
def insert_newlines(string, length=32, lower=True):
    lines = []
    for i in range(0, len(string), length):
        lines.append(string[i:i+length])
    if lower:
        # return ('\n'.join(lines)).lower()     # required with html
        return ('\\n'.join(lines)).lower()    # required with nohtml
    # return '\n'.join(lines)                   # required with html
    return '\\n'.join(lines)                  # required with nohtml


# map bbox center coordinates from image coordinates to page coordinates
def get_scaled_center(img_center_x, img_center_y, img_width, img_height, img_scale, bb_left, bb_bot, bb_right, bb_top):
    bb_center_x = img_center_x - img_width/2 + img_scale * (bb_left + (bb_right - bb_left) / 2)
    bb_center_y = img_center_y - img_height/2 + img_scale * (bb_bot + (bb_top - bb_bot) / 2)
    pos=str(bb_center_x) + "," + str(bb_center_y) + "!"
    return pos


# create action graph visualization from json format description
def json_to_viz(parsed_json, debug=False):
    # graph dimensions (adjust these values)
    is_top_down = True          # grow graph downward
    g_left = 0                  # left boundary of page
    # g_bot = 0                 # bottom boundary of page
    g_horizontal_gap = 1
    g_vertical_gap = 1

    g_txt_width = 5
    g_txt_height = 0.5
    g_txt_border = 0.125        # padding around each side

    g_img_width = 4
    g_img_height = 3

    # g_inner_txt_boundary_color = 'invis'  # required for html
    g_inner_txt_boundary_color = 'gray'
    g_outer_txt_boundary_color = 'dimgray'
    g_txt_font_color = 'midnightblue'

    g_bb_color = 'limegreen'

    g_edge_to_bb_color = 'limegreen'
    g_edge_to_step_color = 'darkorange'

    # resulting attributes
    g_item_height = g_vertical_gap + max(g_txt_height+2*g_txt_border, g_img_height)

    g_txt_center_x = g_left + g_img_width + g_horizontal_gap + g_txt_border + g_txt_width/2
    g_img_center_x = g_left + g_img_width/2

    g_outer_txt_width = g_txt_width + 2*g_txt_border
    g_outer_txt_height = g_txt_height + 2*g_txt_border

    # create graph
    g = Digraph('text_core', engine='neato')

    # set graph and node attributes
    g.graph_attr['splines'] = 'curved'

    g.node_attr['shape']='rect'
    g.node_attr['fixedsize']='true'
    g.node_attr['fontsize'] = '12'
    g.node_attr['fontcolor'] = str(g_txt_font_color)
    g.node_attr['imagescale'] = 'true'
    orientation_top = ':n' if is_top_down else ':s'
    orientation_bot = ':s' if is_top_down else ':n'


    step_count = len(parsed_json)

    # draw nodes for images and texts
    for i in range(step_count):
        # item_id is used for vertical positioning
        item_id = (step_count-1-i) if is_top_down else i

        # formulate nohtml for text node
        text = "<f0>" + insert_newlines(parsed_json[i]['pred'])
        entity_count = len(parsed_json[i]['entities'])
        for j in range(entity_count):
            text += "|<f" + str(j+1) + ">"
            text += insert_newlines(parsed_json[i]['entities'][j])

        # # formulate html for text node
        # text = '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1"><TR><TD PORT="f0">''' + insert_newlines(parsed_json[i]['pred']) + '''</TD>'''
        # entity_count = len(parsed_json[i]['entities'])
        # for j in range(entity_count):
        #     text += '''<TD BGCOLOR="#EEFFDD" PORT="f''' + str(j+1) + '''">''' + insert_newlines(parsed_json[i]['entities'][j]) + '''</TD>'''
        # text += '''</TR></TABLE>>'''

        # draw text node
        # g.node("text_"+str(i),text,pos=str(g_txt_center_x)+","+str(i*g_item_height)+"!",width=str(g_txt_width),height=str(g_txt_height),color=str(g_inner_txt_boundary_color))
        
        g.node("text_shell_"+str(i),'',pos=str(g_txt_center_x)+","+str(item_id*g_item_height)+"!",width=str(g_outer_txt_width),height=str(g_outer_txt_height),color=str(g_outer_txt_boundary_color))
        g.node("text_"+str(i),nohtml(text),shape='Mrecord',pos=str(g_txt_center_x)+","+str(item_id*g_item_height)+"!",width=str(g_txt_width),height=str(g_txt_height),color=str(g_inner_txt_boundary_color))    # required with nohtml
        
        # draw image node
        g.node("img_"+str(i),pos=str(g_img_center_x)+","+str(item_id*g_item_height)+"!",width=str(g_img_width),height=str(g_img_height),label='',image=parsed_json[i]['img'],color='invis')
        
        # draw text-image pairing edge (needed for connected graph to manage overlapping)
        g.edge("text_"+str(i), "img_"+str(i), color='invis')

    # draw edges from entity to action step
    for i in range(step_count):
        entity_count = len(parsed_json[i]['ea'])
        for j in range(entity_count):
            action_id_ref = parsed_json[i]['ea'][j]
            if (action_id_ref != -1):
                g.edge("text_"+str(i)+":f"+str(j+1)+orientation_top,"text_shell_"+str(action_id_ref)+orientation_bot,color=str(g_edge_to_step_color))

    # draw bounding boxes
    for i in range(step_count):
        # item_id is used for vertical positioning
        item_id = (step_count-1-i) if is_top_down else i
        
        bb_count = len(parsed_json[i]['bboxes'])
        width, height = imagesize.get(parsed_json[i]['img'])
        w_ratio = g_img_width/width
        h_ratio = g_img_height/height
        min_ratio = min(w_ratio, h_ratio)

        is_width_bounded = (w_ratio < h_ratio)      # bounded due to width
        w=h_ratio*width if not is_width_bounded else g_img_width
        h=w_ratio*height if is_width_bounded else g_img_height

        for j in range(bb_count):
            bb = parsed_json[i]['bboxes'][j]
            g.node("bb_"+str(i)+"_"+str(j),label='',pos=get_scaled_center(g_img_center_x,item_id*g_item_height,w,h,min_ratio,bb['left'],bb['bot'],bb['right'],bb['top']),width=str((bb['right']-bb['left'])*min_ratio),height=str((bb['top']-bb['bot'])*min_ratio),color=str(g_bb_color))

    # draw edges from entity to bounding boxes
    for i in range(step_count):
        # item_id is used for vertical positioning
        item_id = (step_count-1-i) if is_top_down else i

        edge_orientation = orientation_top if (i < step_count/2) else orientation_bot
        entity_count = len(parsed_json[i]['eb'])
        for j in range(entity_count):
            bb_id_ref = parsed_json[i]['eb'][j]
            if (bb_id_ref != -1):
                g.edge("text_"+str(i)+":f"+str(j+1)+edge_orientation,"bb_"+str(i)+"_"+str(bb_id_ref),color=str(g_edge_to_bb_color))
                # g.edge("text_"+str(i)+":f"+str(j+1)+":n","bb_"+str(i)+"_"+str(bb_id_ref),color=str(g_edge_to_bb_color),n='1',pos="e,152.13,411.67 91.566,463.4 108.12,449.26 127.94,432.34 144.37,418.3")

    if debug:
        print(g.source)

    g.render('visualizer-output/graph')


if __name__ == "__main__":
    if(len(sys.argv) == 2):
        file_to_viz(sys.argv[1])
    else:
    	file_to_viz()


