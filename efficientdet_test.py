
import ast
import os
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

myAnnFileName = 'my_annotations_test.txt'
buses = 'test/'

def change_bbox_format(x1, y1, x2, y2):
    """
    change the format to fit our requierment for x1, y1, width, height
    """
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    w = abs(x2-x1)
    h = abs(y2-y1)
    return int(x_min), int(y_min), int(w), int(h)


compound_coef = 3
force_input_size = None  # set None to use default size
#was 'test/bus_test.jpg'
#img_path = 'test/bus_test.jpg'
# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.7

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['Green Bus', 'Yellow Bus', 'White Bus', 'Grey Bus', 'Blue Bus', 'Red Bus']


color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size


img_names = os.listdir(buses)
img_path = [str(os.path.join(buses, img_name)) for img_name in img_names]
ori_imgs, framed_imgs, framed_metas = preprocess(*img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
# model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}_499_22500.pth', map_location='cpu'))

model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)

    ## out is a list of dicts, where the first item matches the first image that was sent in img_path. each dict has 3 keys:
    ## rois: values are array of (num_of,boxes,4) floats for the bounding boxes, class_ids: values are array of (num_of_boxes,1) holding the class id
    ## matching to the BB box. scores: values are array of (num_of_boxes,1) holding the score for each of that bounding box
    ##important note- the BB_box is of the form [x1, y1, x2, y2]. we need a different form for the annotations file.


annFileEstimations = open(myAnnFileName, 'w+') #create the annotations file to write our model's results on testing images
strToWrite = ''
for imNum, imName in enumerate(img_names):
    strToWrite += imName + ':'
    for box_num, box in enumerate(out[imNum]['rois']):
        x1, y1, w, h = change_bbox_format(*box)
        class_id = out[imNum]['class_ids'][box_num]
        strToWrite+='['+str(x1)+','+str(y1)+','+str(w)+','+str(h)+','+str(class_id+1)+']'
        if box_num != out[imNum]['rois'].shape[0]-1:  # as long as this image contains more boxes, they should be separated by commas (,)
            strToWrite+=','
    strToWrite += '\n' # once finished going over the image boxes we move to the next one in a new line
annFileEstimations.write(strToWrite)
annFileEstimations.close()

# root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# file_dir = 'python_scripts'
# file_name = 'my_annotations_train.txt'



def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


out = invert_affine(framed_metas, out) # todo: why do we need this function- maybe we should work with out post this function
display(out, ori_imgs, imshow=False, imwrite=False)









print('running speed test...')
with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring image for 10 times...')
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors = model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        out = invert_affine(framed_metas, out)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    # uncomment this if you want a extreme fps test
    # print('test2: model inferring only')
    # print('inferring images for batch_size 32 for 10 times...')
    # t1 = time.time()
    # x = torch.cat([x] * 32, 0)
    # for _ in range(10):
    #     _, regression, classification, anchors = model(x)
    #
    # t2 = time.time()
    # tact_time = (t2 - t1) / 10
    # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
