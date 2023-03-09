rm -rf datasets/cocosplit/datasplit/trainvalno5k.json &&
cp -r datasets/cocosplit/datasplit/trainvalno5k-origin.json datasets/cocosplit/datasplit/trainvalno5k.json &&
rm -rf datasets/coco/trainval2014_1 &&
rm -rf datasets/coco/trainval2014_gen &&
rm -rf datasets/cocosplit/seed0 &&
cp -r datasets/cocosplit/seed0-origin datasets/cocosplit/seed0 #&&
#rm -rf data/few_shot_ann/voc/benchmark_1shot 
