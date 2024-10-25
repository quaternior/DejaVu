for l in $(seq 0 2 64)
do  
    (trap 'kill 0' SIGINT; \
    CUDA_VISIBLE_DEVICES=0 python main_mlp.py --dataset c4 --lr 0.001 --L ${l} > logs/c4_mlp_out_${l}.txt & \
    CUDA_VISIBLE_DEVICES=1 python main_mlp.py --dataset c4 --lr 0.001 --L $((l+1)) > logs/c4_mlp_out_$((l+1)).txt & \
    wait)
done