#(jhkim) Added to be github
PATH_TO_MODEL_CHECKPOINT=./pretrained_models/facebook/opt-1.3b
file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_c4_val_opt_1_3b.jsonl
eval_file=./c4_val/eval_c4_val_opt_1_3b.txt
    
echo "start running ${file}"

ARGS="--model-name $PATH_TO_MODEL_CHECKPOINT \
--model-type opt \
--seed 42 \
--fp16 \
--num-layers 12 \
--max-layers 24 \
--budget 22800 \
--num-iters 2000 \
--dist-url tcp://127.0.0.1:9032 \
--token-micro-batch-size 2 \
--world-size 2 --pipeline-group-size 2 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
wait)

python -c "import json
import numpy as np

logprobs = []

with open('$output_file') as f:
    for line in f:
        if line.strip() == '':
            continue
        if 'result' not in json.loads(line):
            break
        item = json.loads(line)

        logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
mean_logprob = sum(logprobs) / len(logprobs)
perplexity = np.exp(-mean_logprob)
print('perplexity:', perplexity)" > $eval_file
cat $eval_file