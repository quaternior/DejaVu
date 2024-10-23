
file=./c4_train/c4_train.json
    
echo "start running ${file}"

ARGS="--model-name facebook/opt-1.3b \
--model-type opt-save \
--seed 42 \
--fp16 \
--num-layers 12 \
--max-layers 96 \
--budget 22800 \
--num-iters 2000 \
--dist-url tcp://127.0.0.1:9032 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file}"

python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0