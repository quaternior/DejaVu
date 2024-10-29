# (jhkim)
file=./c4_train/c4_train.jsonl
    
echo "start running ${file}"

ARGS="--model-name facebook/opt-1.3b \
--save-path ./pretrained_models/facebook/opt-1.3b "

python convert_opt_checkpoint.py $(echo ${ARGS})
    