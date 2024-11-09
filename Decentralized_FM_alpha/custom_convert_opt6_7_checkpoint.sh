# (jhkim)
file=./c4_train/c4_train.jsonl
    
echo "start running ${file}"

ARGS="--model-name facebook/opt-6.7b \
--save-path ./pretrained_models/facebook/opt-6.7b "

python convert_opt_checkpoint.py $(echo ${ARGS})
    