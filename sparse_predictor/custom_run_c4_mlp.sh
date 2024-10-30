for l in $(seq 0 2 22)
do  
    (trap 'kill 0' SIGINT; \
    CUDA_VISIBLE_DEVICES=0 python main_mlp.py --model 1.3b --dataset c4 --lr 0.001 --L ${l} > logs/c4_mlp_out_${l}.txt & \
    CUDA_VISIBLE_DEVICES=1 python main_mlp.py --model 1.3b --dataset c4 --lr 0.001 --L $((l+1)) > logs/c4_mlp_out_$((l+1)).txt & \
    wait)
done
