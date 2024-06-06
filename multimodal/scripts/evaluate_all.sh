
DATASET_DIR="<your dataset path>"

# Define an array of model, dataset
declare -a combinations=(
    "llava-v1.5-7b-f16 vqav2"
    "llava-v1.5-7b-f16 viswiz"
    "llava-v1.5-7b-f16 gqa"
    "llava-v1.5-7b-f16 scienceqa"
    "llava-v1.5-7b-f16 textvqa"
    "llava-v1.5-7b-q8-0 vqav2"
    "llava-v1.5-7b-q8-0 viswiz"
    "llava-v1.5-7b-q8-0 gqa"
    "llava-v1.5-7b-q8-0 scienceqa"
    "llava-v1.5-7b-q8-0 textvqa"
    "llava-v1.5-7b-q4-0 vqav2"
    "llava-v1.5-7b-q4-0 viswiz"
    "llava-v1.5-7b-q4-0 gqa"
    "llava-v1.5-7b-q4-0 scienceqa"
    "llava-v1.5-7b-q4-0 textvqa"
    ...
)

for combo in "${combinations[@]}"; do
    read -r model dataset down_sample <<< "$combo"
    sh mobile_lm/scripts/evaluate_vqa.sh $model $dataset $down_sample
done
