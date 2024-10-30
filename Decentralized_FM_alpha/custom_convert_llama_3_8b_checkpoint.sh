# (jhkim)
file=./c4_train/c4_train.jsonl
    
echo "start running ${file}"

ARGS="--model-name meta-llama/Meta-Llama-3-8B \
--save-path ./pretrained/meta-llama/Meta-Llama-3-8B "

python convert_llama_checkpoint.py $(echo ${ARGS})
    