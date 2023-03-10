EXPNAME=mfdc
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=weight/R-101.pkl                           
IMAGENET_PRETRAIN_TORCH=weight/resnet101-5d3b4d8f.pth  

#CUDA_VISIBLE_DEVICES=4,5,6,7 
#python3 train_net_coco.py --num-gpus 8 --config-file configs/coco/base.yaml     \
#    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
#           OUTPUT_DIR ${SAVEDIR}/base


#python3 tools/model_surgery.py --dataset coco --method randinit                        \
#    --src-path ${SAVEDIR}/base/model_final.pth                         \
#    --save-dir ${SAVEDIR}/base

BASE_WEIGHT=${SAVEDIR}/base/model_reset_surgery.pth


for seed in 0 
do
    for shot in 5 10 30
    do
        sh restart_coco.sh &&
        python read_xml_gennovel_on_rand_base_coco.py -s $shot -n 20 &&
        sh after_coco.sh &&
        cp defrcn/evaluation/coco_evaluation_clip-1.py defrcn/evaluation/coco_evaluation.py &&
        python3 tools/create_config.py --dataset coco14 --config_root configs/coco --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/coco/mfdc_gfsod_novel_${shot}shot_seed${seed}.yaml 
        OUTPUT_DIR=${SAVEDIR}/mfdc_gfsod_novel/tfa-like/${shot}shot_seed${seed}
        python3 train_net_coco.py --num-gpus 4 --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
