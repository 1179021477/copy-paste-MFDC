#coding : utf-8
import os
import cv2
import abc
import math
import matplotlib.pyplot as plt
from PIL import Image

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import copy
import json

'''
read voc xml
'''
COCO_NOVEL_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
]
from pycocotools.coco import COCO 
 

 
 
def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='subcommand for getting frames')
    parser.add_argument('-s', help='shot') #1 to 10
    parser.add_argument('-n',help='select_num') #1 to 200
    #parser.add_argument('-f',help='list file') #none a a1 one one1 real
    #parser.add_argument('-sp',help='split')
    args = parser.parse_args()
    print(args)

    #class_name = ['bird', 'bus', 'cow', 'motorbike', 'sofa'] #1
    #class_name = ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'] #2
    class_label = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72] 
    class_name = []
    for i in range(len(class_label)):
        class_name.append(COCO_NOVEL_CATEGORIES[i]["name"])
    print("classes:",class_name)

    #data_root = "data/coco/trainval2014/"
    #save_img_root = "data/coco/trainval2014_gen/"
    #save_anno_root = "data/few_shot_ann/coco/annotations/trainvalno5k.json"
    data_root = "datasets/coco/trainval2014/"
    save_img_root = "datasets/coco/trainval2014_gen/"
    save_anno_root = "datasets/cocosplit/datasplit/trainvalno5k.json"

    shot = args.s+'shot'
    select_num = int(args.n)
    img_file = "/nvme/nvme2/linshaobo/text_to_img_generator/sd_save_COCO_imagenetprompts_a_200_newrect_cutout/"
    files = os.listdir(img_file)
    coco_base_file = "trainval_base.json"
    #base_img_list = "base_list/select_base_coco_"+str(shot)+"_0.6_10.txt"
    #coco = COCO(save_anno_root)

    novel_list_all = []
    for name in files:# novel 
          img = cv2.imread(img_file+name)
          for cl in range(len(class_name)):
               if name.split("_")[0] in class_name[cl]: #novel
                   for i in range(len(COCO_NOVEL_CATEGORIES)):
                        if COCO_NOVEL_CATEGORIES[i]["name"] == class_name[cl]:
                             novel_list_all.append({"1":img,"2":COCO_NOVEL_CATEGORIES[i]["id"],"4":name})
                             #print("label:",class_name[cl])
                             break
                   break
                      
    base_coco = COCO(coco_base_file) 
    f_base = open(coco_base_file)
    lines = f_base.readlines()
    base_info = json.loads(lines[0])
    base_images = base_info["images"]
    base_annos = base_info["annotations"]
    count = 0
    base_list_all = []
    
    for img in base_images:
         base_bboxes_all = [] 
         base_img_name = img["file_name"]
         img_id = img["id"]
         base_img = cv2.imread(data_root + base_img_name)                 
         anns_id = base_coco.getAnnIds(imgIds=img_id)
         anns = base_coco.loadAnns(anns_id)
         for i in range(len(anns)):                     
            base_bboxes_all.append([int(anns[i]["bbox"][0]),int(anns[i]["bbox"][1]),int(anns[i]["bbox"][0]+anns[i]["bbox"][2]),
            int(anns[i]["bbox"][1]+anns[i]["bbox"][3])])
           
         base_list_all.append({"base_img":base_img,"base_bboxes_all":base_bboxes_all,"base_img_name":base_img_name}) 
         count += 1

    label = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72] 
    dict_num = {} 
    for i in range(len(label)):
        dict_num[str(label[i])] = 0
   
    imgs_info_list = []
    annos_info_list = []

    id_index = 0
    count_k = 0
    for k in range(len(novel_list_all)):
            novel_instances = novel_list_all[k]["1"]
            novel_labels_list = novel_list_all[k]["2"]
            name = novel_list_all[k]["4"]
 
            if not os.path.exists(save_img_root):
              os.makedirs(save_img_root)

         # resized instance ratio of base img           
            object_ratio = random.uniform(3,8)            
            tr = novel_labels_list

            if dict_num[str(tr)] == select_num:
                 continue
            print("dict-num:",dict_num)
            dict_num[str(tr)] += 1 
            count_k += 1
            base_img =  copy.deepcopy(base_list_all[k]["base_img"])    
            base_img_name = base_list_all[k]["base_img_name"]
            base_bboxes_all = base_list_all[k]["base_bboxes_all"]
            H,W,C = base_img.shape              
 
            H1,W1,C1 = novel_instances.shape
            wh_ratio = float(W1)/float(H1)
            hw_ratio = float(H1)/float(W1)
 
           #avoid filling the base instances: by gen 100 box find min sum of iou novel box with  all base boxes
            all_novel_list = []
            novel_bboxs_list = []
            tmp_list = []

            if W >= H:
              RATIO = wh_ratio
            else:
              RATIO = hw_ratio
            flag_ex = False
            for k in range(1000):
                     # origin wh ratio
               try:
                     if W >= H:
                         tmp = cv2.resize(novel_instances,(int(W/object_ratio),int(W/object_ratio/RATIO))) # how to resize:keep novel ratio or others
                         #tmp_mask = cv2.resize(mask_img,(int(W/object_ratio),int(W/object_ratio/RATIO)))
                         if H < int(W/object_ratio/RATIO):
                              object_ratio = W/(H*RATIO)+1
                         loc_xmin = random.randint(1,W-int(W/object_ratio))
                         loc_ymin = random.randint(1,H-int(W/object_ratio/RATIO))
                         w_o = tmp.shape[1] #int(W/object_ratio)
                         h_o = tmp.shape[0] 
                     else:
                         tmp = cv2.resize(novel_instances,(int(H/object_ratio/RATIO),int(H/object_ratio))) # how to resize:keep novel ratio or others
                         #tmp_mask = cv2.resize(mask_img,(int(H/object_ratio/RATIO),int(H/object_ratio)))
                         if W < int(H/object_ratio/RATIO):
                             object_ratio = H/(W*RATIO)+1
                         loc_xmin = random.randint(1,W-int(H/object_ratio/RATIO))
                         loc_ymin = random.randint(1,H-int(H/object_ratio))
                         w_o = tmp.shape[1] #int(W/object_ratio)
                         h_o = tmp.shape[0]
 
                     #### base wh ratio
                     #tmp = cv2.resize(novel_instances,(int(W/object_ratio),int(H/object_ratio))) # how to resize:keep novel ratio or others
                     #loc_xmin = random.randint(1,W-int(W/object_ratio))
                     #loc_ymin = random.randint(1,H-int(H/object_ratio))
                     #w_o = tmp.shape[1] #int(W/object_ratio)
                     #h_o = tmp.shape[0] 

                     bboxes = [loc_ymin,loc_xmin,loc_ymin+h_o,loc_xmin+w_o]
                     novel_bboxs_list.append([loc_xmin,loc_ymin,loc_xmin+w_o,loc_ymin+h_o])
                     all_novel_list.append(bboxes)
                     tmp_list.append(tmp)
               except:
                 flag_ex = True
                 break
            if flag_ex:
                 print("1111")
                 continue

            iou_list = []
            for m in range(len(all_novel_list)):
                   rec1 = all_novel_list[m]
                   iou = 0
                   for j in range(len(base_bboxes_all)):  
                        [b_xmin,b_ymin,b_xmax,b_ymax] = base_bboxes_all[j]
                        rec2 = [b_ymin,b_xmin,b_ymax,b_xmax]
                        iou  += compute_iou(rec1, rec2)  
                   iou_list.append(iou) 
            min_index = np.argmin(np.array(iou_list))                     
            tmp = tmp_list[min_index]

            novel_bbox_list = novel_bboxs_list[min_index]
            loc_ymin = novel_bbox_list[1]
            loc_xmin = novel_bbox_list[0]
            ymax = novel_bbox_list[3]
            xmax = novel_bbox_list[2]
            try:
                base_img[loc_ymin:ymax,loc_xmin:xmax] = tmp
            except:
                continue
            save_img_name = base_img_name.split('.')[0]+name.split('.')[0]+"_"+str(count_k)+".jpg"
            cv2.imwrite(save_img_root+save_img_name,base_img)
 
            id_index += 1
            image_info = {"license": 2, "file_name": save_img_name, "coco_url": None, "height": H, "width": W, "date_captured": "2022-09-22", "flickr_url": None, "id": int(str(id_index)+"000000")}
            anno_info = {"segmentation": [[]], "area": None, "iscrowd": 0, "image_id": int(str(id_index)+"000000"), "bbox": [loc_xmin, loc_ymin, xmax-loc_xmin,ymax-loc_ymin], "category_id": tr, "id": int(str(id_index)+"000000")}

            imgs_info_list.append(image_info)
            annos_info_list.append(anno_info)  
    
            #shot json files
            train_json = "datasets/cocosplit/seed0/full_box_"+str(shot)+"_"+class_name[class_label.index(tr)]+"_trainval.json"
            print("shot json:",train_json)
            f_anno = open(train_json,"r")
            annos_1 = json.load(f_anno)
            annos_1["images"].append(image_info)
            annos_1["annotations"].append(anno_info)
            f_1 = open(train_json,"w")
            f_1.write(json.dumps(annos_1))
            
            #dict_num[class_label.index(tr)] += 1
            #break

    # save new annos on trainvalno5k.json
    f_anno = open(save_anno_root,"r")
    annos_ = json.load(f_anno)

    for k in range(len(imgs_info_list)):
            #print("f_anno:",len(annos_["images"]),len(annos_["images"]))
            annos_["images"].append(imgs_info_list[k])
            annos_["annotations"].append(annos_info_list[k])       
    f_save = open(save_anno_root,"w")
    f_save.write(json.dumps(annos_))

