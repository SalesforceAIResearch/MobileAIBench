python multimodal/evaluate_vqa.py \
    --seed 0 \
    --multimodal \
    --model ${1} \
    --dataset ${2} \
    --model_config_path multimodal/config/model_config.yaml\
    --dataset_dir ... \
    --down_sample 1000