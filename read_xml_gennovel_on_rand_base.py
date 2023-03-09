#coding : utf-8
import os
import cv2
import abc
import xml.dom.minidom as xml
import math
import matplotlib.pyplot as plt
from PIL import Image

from save_xml import save_xml 
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import copy

#import aug_lib


#from autoaugment import ImageNetPolicy
#from autoaugment import CIFAR10Policy
#from autoaugment import SVHNPolicy

'''
read voc xml
'''
 

 
class XmlReader(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass
    def read_content(self,filename):
        content = None
        if (False == os.path.exists(filename)):
            return content
        filehandle = None
        try:
            filehandle = open(filename,'rb')
        except FileNotFoundError as e:
            print(e.strerror)
        try:
            content = filehandle.read()
        except IOError as e:
            print(e.strerror)
        if (None != filehandle):
            filehandle.close()
        if(None != content):
            return content.decode("utf-8","ignore")
        return content
 
    @abc.abstractmethod
    def load(self,filename):
        pass
 
class XmlTester(XmlReader):
    def __init__(self):
        XmlReader.__init__(self)
    def load(self, filename):
        filecontent = XmlReader.read_content(self,filename)
        if None != filecontent:
            dom = xml.parseString(filecontent)
            root = dom.getElementsByTagName('annotation')[0]
            #im_size = root.getElementsByTagName('size')[0]
 
            #im_w = int((im_size.getElementsByTagName('width')[0]).childNodes[0].data)
            #im_h = int((im_size.getElementsByTagName("height")[0]).childNodes[0].data)
            #im_shape=[im_w,im_h]
            #print(dom.getElementsByTagName('object'))
            len_objs = len(dom.getElementsByTagName('object'))
            #print("len:",len_objs)
            labels = []
            diffs = []
            bboxs = []
            for i in range(len_objs):
                obj = dom.getElementsByTagName('object')[i]
                box = obj.getElementsByTagName('bndbox')[0]
                #print(obj)
                label = str((obj.getElementsByTagName("name")[0]).childNodes[0].data)
                diff = int((obj.getElementsByTagName("difficult")[0]).childNodes[0].data)
                labels.append(label)
                diffs.append(diff)

                b_xmin=int((box.getElementsByTagName("xmin")[0]).childNodes[0].data)
                b_ymin=int((box.getElementsByTagName("ymin")[0]).childNodes[0].data)
                b_xmax=int((box.getElementsByTagName("xmax")[0]).childNodes[0].data)
                b_ymax=int((box.getElementsByTagName("ymax")[0]).childNodes[0].data)
 
                bbox=[b_xmin,b_ymin,b_xmax,b_ymax]
                bboxs.append(bbox)
  
            return labels, diffs, bboxs
 
 
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
 
 #aug tricks for novel instances
"""Randomly mask out one or more patches from an image."""
def Cutout(img, n_holes=2, length=32, prob=0.5):
     if np.random.rand() < prob:
            h = img.size(1)
            w = img.size(2)
            mask = np.ones((h, w), np.float32)
            for n in range(n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)
                mask[y1:y2, x1:x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

     return img
        
        
trans = transforms.Compose(
      [ 
        #transforms.RandomCrop(32, padding=4),  #先四周填充0，然后图像随机裁剪成32*32
        #transforms.Resize(),
        #transforms.CenterCrop(),
        #transforms.RandomResizedCrop(),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.1),
        #transforms.RandomRotation(10),
        #ImageNetPolicy(),#auto-augment
        #SVHNPolicy(),
        #CIFAR10Policy(),
        #transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.1),
       ]
)


 
 
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='subcommand for getting frames')
    parser.add_argument('-s', help='shot') #1 to 10
    parser.add_argument('-n',help='select_num') #1 to 200
    parser.add_argument('-f',help='list file') #none a a1 one one1 real
    parser.add_argument('-sp',help='split')
    args = parser.parse_args()
    print(args)

    shot = args.s +"shot"
    select_num = int(args.n)
    last_file = "sd_save_VOC_split"+args.sp+"_imagenetprompts_none_200_newrect_cutout/" #_remove/"
    #last_file = "sd_save_VOC_split1_imagenetprompts_none_200_cutout/"
    #last_file = "VOC_split1_imagenetprompts_none_200/"

    last_file = last_file.replace("none",args.f)
    if int(args.sp) == 1: 
        class_name = ['bird', 'bus', 'cow', 'motorbike', 'sofa'] #1
    if int(args.sp) == 2:
        class_name = ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'] #2
    if int(args.sp) == 3:
        class_name = ['boat', 'cat', 'motorbike', 'sheep', 'sofa'] #3
    data_root = "datasets/" #"data/VOCdevkit/"
    #shot = '3shot'
    #select_num = 200#100#10
    #last_file = "sd_save_VOC_split1_imagenetprompts_none_200_newrect_cutout_remove/"
    list_file = "/nvme/nvme2/linshaobo/text_to_img_generator/" + last_file
    #list_file = "/nvme/nvme2/linshaobo/text_to_img_generator/sd_save_VOC_split1_imgtoimg_fp_0.5_50_s0.95_imagenetprompts8_1_cutout_remove/" #_origin/" #cutout_remove/"
    base_img_list = "datasets/VOC2007/ImageSets/Main/test_VOC2007_split"+args.sp+".txt"
    files = os.listdir(list_file)
    reader = XmlTester()
    
    #augmenter = aug_lib.TrivialAugment()  

    novel_list_all = []
    for name in files:# novel 
              img = cv2.imread(list_file+name)
              for cl in range(len(class_name)):
                   if class_name[cl] in name: #novel                      
                       #novel_instances.append(img)
                       #novel_labels_list.append(class_name[cl])
                       novel_list_all.append({"1":img,"2":class_name[cl],"3":"VOC2007","4":name,"5":0})
                       #print("label:",labels[cl])

    temo_list = novel_list_all
    novel_list_all = []
                        
    for i in range(len(class_name)):
        for j in range(len(temo_list)):
            if temo_list[j]["2"] == class_name[i]:
                 novel_list_all.append(temo_list[j])

    print(len(novel_list_all))           
           #base img list
    f_base = open(base_img_list,"r")
    lines_base = f_base.readlines()

     # reduce chongfu base
    count = 0
    base_list_all = []
    for line_base in lines_base:
         #print("count:",count,line_base)
         count += 1
         base_img_name = line_base.rstrip()+".jpg"
         base_path = data_root + "VOC2007/JPEGImages/"+base_img_name
         #print(base_path)
         base_img = cv2.imread(base_path)
         base_xml_path = data_root + "VOC2007/Annotations/"+base_img_name.replace(".jpg",".xml")
                    
         #H,W,C = base_img.shape    
 
         base_labels_all, base_diff_all, base_bboxes_all = reader.load(base_xml_path)
         base_list_all.append({"1":base_img,"2":base_bboxes_all,"3":base_xml_path,"4":base_img_name}) 

    dict_num = {"15":0,"16":0,"17":0,"18":0,"19":0}

    count_k = 0

    for k in range(len(novel_list_all)):
         #print(k)
         novel_instances = novel_list_all[k]["1"]
         novel_label_list = novel_list_all[k]["2"]
         pre_fix = novel_list_all[k]["3"]
         name = novel_list_all[k]["4"]
         diffs = novel_list_all[k]["5"]
         save_root = data_root + pre_fix+"/Annotations_1/"
         save_img_root = data_root + pre_fix+"/JPEGImages_1/"
         if not os.path.exists(save_root):
              os.makedirs(save_root)
         if not os.path.exists(save_img_root):
              os.makedirs(save_img_root)

         # resized instance ratio of base img           
         object_ratio = random.uniform(3,8)           
         #print(diffs) 
         if int(diffs) == 0: 
            tr = class_name.index(novel_label_list)+15
            #print(tr)
            # max number of novel 
            if dict_num[str(tr)] == select_num:
               continue
              
            dict_num[str(tr)] += 1
            count_k += 1
            print("dict-num:",dict_num)
            import copy
            #if k1 > len(dict_b[str(tr)]):
            #    k1 = 0
            #print(k1,len(dict_b[str(tr)]))
           
            base_list_all
            base_img = copy.deepcopy(base_list_all[count_k]["1"])    
            base_bboxes_all =  base_list_all[count_k]["2"] 
            base_xml_path = base_list_all[count_k]["3"]
            base_img_name = base_list_all[count_k]["4"]
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
            labels_base,diffs_base,bboxs_base=reader.load(base_xml_path)
            #print("###",novel_label_list,novel_bbox_list)
            save_xml(W,H,labels_base,diffs_base,bboxs_base,save_img_name,save_root, novel_label_list, novel_bbox_list) 
            base_img = None
            #break
 
