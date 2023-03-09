rm -rf datasets/VOC2007/Annotations datasets/VOC2007/JPEGImages &&
cp -r ../mmfewshot/data/VOCdevkit/VOC2007/Annotations-origin datasets/VOC2007/Annotations &&
cp -r datasets/VOC2007/Annotations_1/* datasets/VOC2007/Annotations/ &&
cp -r ../mmfewshot/data/VOCdevkit/VOC2007/JPEGImages-origin datasets/VOC2007/JPEGImages &&
cp -r datasets/VOC2007/JPEGImages_1/* datasets/VOC2007/JPEGImages/ &&
rm -rf datasets/VOC2012/Annotations datasets/VOC2012/JPEGImages &&
cp -r ../mmfewshot/data/VOCdevkit/VOC2012/Annotations-origin datasets/VOC2012/Annotations &&
#cp -r datasets/VOC2012/Annotations_1/* datasets/VOC2012/Annotations/ &&
cp -r ../mmfewshot/data/VOCdevkit/VOC2012/JPEGImages-origin datasets/VOC2012/JPEGImages &&
#cp -r  datasets/VOC2012/JPEGImages_1/* datasets/VOC2012/JPEGImages/ &&
cp -r datasets/VOC2007/JPEGImages_1/* datasets/VOC2012/JPEGImages/ && # appendix
#cp -r datasets/VOC2012/Annotations_1/* datasets/VOC2007/Annotations/ &&
#cp -r datasets/VOC2012/JPEGImages_1/* datasets/VOC2007/JPEGImages/ &&
cp -r datasets/VOC2007/Annotations_1/* datasets/VOC2012/Annotations/ 
