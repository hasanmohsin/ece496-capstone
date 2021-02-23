import matplotlib.pyplot as plt
import json
from visual_ground import *

model = VL_Model()

def load_meccano_json(dataset_type = 'train'):
    if dataset_type = 'val':
        path = '../../MECCANO/home/fragusa/instances_meccano_val.json'
    elif dataset_type = 'test':
        path = '../../MECCANO/home/fragusa/instances_meccano_test.json'
    else:
        path = '../../MECCANO/home/fragusa/instances_meccano_train.json'
    
    with open(path) as json_file:
            meccano_json = json.load(json_file)

    return meccano_json

#returns image, bbox, and category
def meccano_json_extract_sample(meccano_json, annot_id=None):
    #if no id passed, sample random one from set
    if annot_id is None:
        annot_id = np.random.randint(0, len(meccano_json['annotations'])-1)
    
    annot_info = meccano_json['annotations'][annot_id]
    im_id = annot_info['image_id']
    im_info = meccano_json['images'][im_id]
    category_info = meccano_json['categories'][annot_info['category_id']]
    
    im_path = im_info['file_name']
    bbox = annot_info['bbox']
    category = category_info['name']
    
    #bbox coords are in format (x,y, width height)
    #change them to (x1, y1, x2, y2) format
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
    
    return '../../MECCANO/active_object_frames/'+im_path, bbox, category




def compute_topN_iou(im_path, gt_bbox_coords, N = 15):
    img = Image.open(im_path)
    im = np.array(img).astype(np.float32)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    
    #get predicted bboxes
    pred_bbox_list = model.detectron_get_bbox(path)[:N].cpu().detach().numpy()
   
    #convert gt bbox coords to same scale
    for coord in gt_bbox_coords:
        coord *= im_scale
    
    best_iou = 0.0
    
    for pred_bbox in pred_bbox_list:
        iou = compute_iou(pred_bbox, gt_bbox_coords)
        
        if iou > best_iou:
            best_iou = iou
            
    return best_iou
     
def compute_iou(pred_bbox, gt_bbox):
    #bottom left corner
    pred_x1 = pred_bbox[0]
    pred_y1 = pred_bbox[1]
    
    #top right corner
    pred_x2 = pred_bbox[2]
    pred_y2 = pred_bbox[3]
    
    gt_x1 = gt_bbox[0]
    gt_y1 = gt_bbox[1]
    gt_x2 = gt_bbox[2]
    gt_y2 = gt_bbox[3]
    
    #coordinates of intersected box
    int_x1 = max(pred_x1, gt_x1)
    int_y1 = max(pred_y1, gt_y1)
    int_x2 = min(pred_x2, gt_x2)
    int_y2 = min(pred_y2, gt_y2)
    
    if int_x1 > int_x2 or int_y1 > int_y2:
        return 0.0
    
    int_area = (int_y2 - int_y1)*(int_x2 - int_x1)
    
    pred_area = (pred_x2 - pred_x1)*(pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1)*(gt_y2 - gt_y1)
    
    union_area = pred_area + gt_area - int_area
    
    iou = int_area/float(union_area)
    
    return iou


#function to evaluate on json, mean iou
def meccano_eval_iou(meccano_json, N=20):
    
    num_samples = len(meccano_json['annotations'])
    
    iou_sum = 0.0
    min_iou = 1.0
    max_iou = 0.0
    
    for annot_id in range(num_samples):
        im_path, bbox, category = meccano_json_extract_sample(meccano_json, annot_id)
        
        iou = compute_topN_iou(im_path, gt_bbox_coords=bbox, N = N)
        
        iou_sum += iou
        
        if iou < min_iou:
            min_iou = iou
        if iou > max_iou:
            max_iou = iou
        
        print("Done---{}/{}".format(annot_id, num_samples))
    mean_iou = iou_sum/float(num_samples)
    
    #return mean_iou
    return mean_iou, min_iou, max_iou

