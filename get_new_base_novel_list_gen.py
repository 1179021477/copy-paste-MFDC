import os
#run twice for
#VOC2007 VOC2012
pre= "VOC2007"
combine_data_path = "datasets/"+pre+"/Annotations_1/"
data_root = "datasets/"
txt_list = "datasets/vocsplit/seed0/"

import argparse
parser = argparse.ArgumentParser(description='subcommand for getting frames')
parser.add_argument('-s', help='shot') #1 to 10
parser.add_argument('-sp', help="split")
args = parser.parse_args()
print(args)
 
shot = args.s +"shot"

if int(args.sp) == 1:
    class_name = ['bird', 'bus', 'cow', 'motorbike', 'sofa'] #1
if int(args.sp) == 2:
    class_name = ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'] #2
if int(args.sp) == 3:
    class_name = ['boat', 'cat', 'motorbike', 'sheep', 'sofa'] #3

if int(args.sp) == 1:
    base_name = ['aeroplane', 'bicycle', 'boat', 'bottle', 'car',
           'cat', 'chair', 'diningtable', 'dog', 'horse', 
           'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
if int(args.sp) == 2:
    base_name = ['bird', 'bicycle', 'boat', 'bus', 'car',
             'cat', 'chair', 'diningtable', 'dog', 'motorbike', 
             'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
if int(args.sp) == 3:
    base_name = ['aeroplane', 'bicycle', 'bird', 'bottle', 'car',
             'bus', 'chair', 'diningtable', 'dog', 'horse', 
             'person', 'pottedplant', 'cow', 'train', 'tvmonitor']


#shot = "3shot"
#class_name = ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'] #2
#class_name = ['boat', 'cat', 'motorbike', 'sheep', 'sofa'] #3
files = os.listdir(txt_list)
base_imgs = os.listdir(combine_data_path)
    
for list_name in files:# novel
    if shot in list_name:
        #print(list_name)
        if class_name[0] in list_name or class_name[1] in list_name or class_name[2] in list_name or class_name[3] in list_name or class_name[4] in list_name:
            #f_novel = open(txt_list+list_name,'r')
            f_novel_save = open(txt_list+list_name,'a')
            for base_img in base_imgs:
                    #print(list_name,base_img)
                    if list_name.split("_")[-2] in base_img:
                        print(list_name,base_img)
                        f_novel_save.write(data_root+'VOC2007/JPEGImages/'+base_img.replace('xml','jpg')+'\n')   
        else:
            f_base = open(txt_list+list_name,"r")
            f_base_save = open(txt_list+list_name,"a")
            lines = f_base.readlines()
            cl = list_name.split("_")[2]
            for base_img in base_imgs:
                xml_file = combine_data_path +"/" +base_img
                if not os.path.exists(xml_file):
                     continue
                else:
                    f = open(xml_file,"r")
                    lines1 = f.readlines()
                    for l in lines1:
                        if cl in l: 
                            f_base_save.write(data_root+pre+'/JPEGImages/'+base_img.replace('xml','jpg')+'\n')
                            break 
