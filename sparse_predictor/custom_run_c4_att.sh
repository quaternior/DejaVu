for l in $(seq 0 2 24)
do  
    (trap 'kill 0' SIGINT; \
    CUDA_VISIBLE_DEVICES=0 python main_att.py --dataset c4 --lr 0.0001 --k 0.3 --L ${l} > ../Decentralized_FM_alpha/logs/c4_att_out_${l}.txt & \
    CUDA_VISIBLE_DEVICES=1 python main_att.py --dataset c4 --lr 0.0001 --k 0.3 --L $((l+1)) > ../Decentralized_FM_alpha/logs/c4_att_out_$((l+1)).txt & \
    wait)
done