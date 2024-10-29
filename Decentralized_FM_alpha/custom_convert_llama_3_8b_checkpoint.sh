# (jhkim)
file=./c4_train/c4_train.jsonl
    
echo "start running ${file}"

ARGS="--model-name pretrained/meta-llama/Meta-Llama-3-8B \
--save-path ./pretrained/meta-llama/Meta-Llama-3-8B "

python convert_opt_checkpoint.py $(echo ${ARGS})
    