from xml.dom.minidom import Document
import os
import json

#label_path = "/mnt/lustre/linshaobo/elevator_object/elevator_transfer/data_elevator/POC_test_data/Luggage/pre_label/"
#img_root_path = "/mnt/lustre/linshaobo/POD_fewshot/data/POC_data/JPEGImages/"
#save_root = "/mnt/lustre/linshaobo/POD_fewshot/data/POC_data/Annotations_1/"

count = 0

tagNames = ['annotation', 'folder', 'filename', 'size', 'object']

def save_xml(W,H,labels,diffs,bboxs,img_name,save_root,labels_new,bboxs_new):
 
    if not os.path.exists(save_root):
       os.makedirs(save_root)
   
    # 创建doc
    doc = Document()
    # 创建根节点
    root = doc.createElement(tagNames[0])
    doc.appendChild(root)

    folder = doc.createElement(tagNames[1])
    folder_text = doc.createTextNode("VOC2007")
    folder.appendChild(folder_text)
    root.appendChild(folder)

    filename = doc.createElement(tagNames[2])
    filename_text = doc.createTextNode(img_name)
    filename.appendChild(filename_text)
    root.appendChild(filename)

    size = doc.createElement(tagNames[3])
    root.appendChild(size)
    width = doc.createElement("width")
    height = doc.createElement("height")
    depth = doc.createElement("depth")
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    w_text = doc.createTextNode(str(W))
    h_text = doc.createTextNode(str(H))
    d_text = doc.createTextNode(str(3))
    width.appendChild(w_text)
    height.appendChild(h_text)
    depth.appendChild(d_text)

    for j in range(len(bboxs)):
        [xmin,ymin,xmax,ymax] = bboxs[j]
        label = str(labels[j])
        diff = str(diffs[j])
 
        object1 = doc.createElement(tagNames[4])
        root.appendChild(object1)
        label_info = doc.createElement("name")
        label_text = doc.createTextNode(label)
        label_info.appendChild(label_text)
        object1.appendChild(label_info)
 
        diff_info = doc.createElement("difficult")
        diff_text = doc.createTextNode(diff)
        diff_info.appendChild(diff_text)
        object1.appendChild(diff_info)
        bndbox_info = doc.createElement("bndbox")
        object1.appendChild(bndbox_info)

        xmin_info = doc.createElement("xmin")
        xmin_text = doc.createTextNode(str(xmin))
        xmin_info.appendChild(xmin_text)
        bndbox_info.appendChild(xmin_info)        
        ymin_info = doc.createElement("ymin")
        ymin_text = doc.createTextNode(str(ymin))
        ymin_info.appendChild(ymin_text)
        bndbox_info.appendChild(ymin_info) 
        xmax_info = doc.createElement("xmax")
        xmax_text = doc.createTextNode(str(xmax))
        xmax_info.appendChild(xmax_text)
        bndbox_info.appendChild(xmax_info) 
        ymax_info = doc.createElement("ymax")
        ymax_text = doc.createTextNode(str(ymax))
        ymax_info.appendChild(ymax_text)
        bndbox_info.appendChild(ymax_info) 
            
    if labels_new:
    #for k in range(len(labels_new)):
        [xmin,ymin,xmax,ymax] = bboxs_new#[k]
        label = labels_new#[k]
        diff = '0'
 
        object1 = doc.createElement(tagNames[4])
        root.appendChild(object1)
        label_info = doc.createElement("name")
        label_text = doc.createTextNode(label)
        label_info.appendChild(label_text)
        object1.appendChild(label_info)
 
        diff_info = doc.createElement("difficult")
        diff_text = doc.createTextNode(diff)
        diff_info.appendChild(diff_text)
        object1.appendChild(diff_info)
        bndbox_info = doc.createElement("bndbox")
        object1.appendChild(bndbox_info)

        xmin_info = doc.createElement("xmin")
        xmin_text = doc.createTextNode(str(xmin))
        xmin_info.appendChild(xmin_text)
        bndbox_info.appendChild(xmin_info)        
        ymin_info = doc.createElement("ymin")
        ymin_text = doc.createTextNode(str(ymin))
        ymin_info.appendChild(ymin_text)
        bndbox_info.appendChild(ymin_info) 
        xmax_info = doc.createElement("xmax")
        xmax_text = doc.createTextNode(str(xmax))
        xmax_info.appendChild(xmax_text)
        bndbox_info.appendChild(xmax_info) 
        ymax_info = doc.createElement("ymax")
        ymax_text = doc.createTextNode(str(ymax))
        ymax_info.appendChild(ymax_text)
        bndbox_info.appendChild(ymax_info)  

    save_path = os.path.join(save_root,img_name.replace(".jpg",".xml").replace(".png",".xml").replace(".jpeg",".xml")) #img_name.split(".")[0]+".xml"
    
    print("save:",count,save_path)
    with open(save_path, 'w') as f:
        f.write(doc.toprettyxml(indent='\t'))
    f.close()

