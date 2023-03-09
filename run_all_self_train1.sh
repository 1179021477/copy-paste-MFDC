#for num in 20 #10 20 50 100 200
#do

num=20
file="a"
echo $num
#sh restart.sh &&
#python read_xml_gennovel_instance_on_rand_base.py -s $shot -n $num -f $file &&
#python get_new_base_novel_list_gen.py -s $shot &&
#sh after_xml.sh &&
#sh restart.sh &&
#sh run_voc-exp-10shot.sh baseline 3 &&

for split in 1
do
for shot in 1 3 5 10
do
#for score in "0.8" #"0.1" "0.2" "0.3" "0.5" #"0.6" "0.7" "0.8" "0.9"
#for mscore in "0" "0.3" "0.4" "0.5" "0.6" "0.8"
#do
echo $shot
echo $score
sh restart.sh &&
python read_xml_gennovel_on_rand_base.py -s $shot -n $num -f $file -sp $split &&
python get_new_base_novel_list_gen.py -s $shot -sp $split &&
sh after_xml.sh &&
cp -r defrcn/evaluation/pascal_voc_evaluation_0.2_split$split.py defrcn/evaluation/pascal_voc_evaluation.py &&
#python self_train_add_novel.py -sh $shot -sc "0.8" -scm "0.5" -sp $split && #revise clip_path and model score
#python self_train_add_novel_after.py -s $shot -sp $split &&
#sh after_xml_self_train.sh &&
##python self_train_add_novel_twostep.py -sh $shot -sc "0.8" -scm "0.5" &&
##python self_train_add_novel_after.py -s $shot &&
##sh after_xml_self_train.sh &&
sh voc_train-exp.sh $split $shot #&&
#mv test_res_clip.txt test_res_clip_trainval2007_${score}_${shot}shot_split${split}.txt
done
done
#done

#sh coco_train.sh
