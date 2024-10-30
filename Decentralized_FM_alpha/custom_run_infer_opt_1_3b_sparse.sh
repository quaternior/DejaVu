#(jhkim)
PATH_TO_MODEL_CHECKPOINT=./pretrained_models/facebook/opt-1.3b
PATH_TO_SPARSITY_PREDICTOR=../checkpoint/opt-1.3b-sparse-predictor
file=./c4_val/c4_valid.jsonl
output_file=./c4_val/output_c4_val_opt_1_3b_sparse.jsonl
eval_file=./c4_val/eval_c4_val_opt_1_3b_sparse.txt

echo "start running ${file}"
# Although these will not be in ARGS, it will be read by `os_environ`
# (Critical variables : SPARSE_PATH, LAYER, SPARSE_ATT)
# LAYER seems like an threshold. (i.e. If layer_index gets bigger than LAYER, disable the predictor)
# TOPK, ATTN_TOPK_1, ATTN_TOPK_2 will do nothing
# SPARSE_ATT works as boolean variable, whether SPARSE_ATT is used or not
export SPRARSE_PATH=$PATH_TO_SPARSITY_PREDICTOR
export LAYER=86
export TOPK=5000
export ATTN_TOPK_1=24
export ATTN_TOPK_2=48
export SPARSE_ATT=1

LAYER=86
TOPK=5000
ATTN_TOPK_1=24
ATTN_TOPK_2=48

ARGS="--model-name $PATH_TO_MODEL_CHECKPOINT \
--model-type opt-ml-att-sparse \
--seed 42 \
--fp16 \
--num-layers 12 \
--max-layers 24 \
--budget 22800 \
--num-iters 2000 \
--dist-url tcp://127.0.0.1:9031 \
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
    n = 0
    for line in f:
        if line.strip() == '':
            continue
        if 'result' not in json.loads(line):
            break
        item = json.loads(line)

        logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
        n += 1
mean_logprob = sum(logprobs) / len(logprobs)
perplexity = np.exp(-mean_logprob)
print('perplexity:', perplexity)" > $eval_file
cat $eval_file

