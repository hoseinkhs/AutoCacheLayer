mkdir "log"
XD=100
BS=16
backbone="Resnet50"
exit_type="Dense2LayerTemp"
exit_epoch=19
trial="50-meters"
experiment="Cifar100"
# --run_server \
# --search_cache_models \
# --exit_on_all_resolved \
    # --count_flops \
python train.py \
    --run_profiler \
    --run_meters \
    --experiment "${experiment}" \
    --num_classes ${XD} \
    --shrink \
    --fine_tune 0 \
    --trial "${trial}" \
    --train_epochs 5 \
    --train_device "cuda:0" \
    --test_device "cuda:0" \
    --exit_model_path "./out_dir/${experiment}/${backbone}/exits/${exit_type}/" \
    --data_root "./data/cifar100" \
    --backbone_type "${backbone}" \
    --backbone_conf_file "backbone_conf.yaml" \
    --exit_type "${exit_type}" \
    --exit_conf_file "./cifar_exit_conf.yaml" \
    --lr 0.1 \
    --out_dir "out_dir/${experiment}/${backbone}" \
    --step "10, 13, 16" \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size $BS \
    --test_batch_size 128 \
    --momentum 0.9 \
    --log_dir "log" \
    --tensorboardx_logdir "mv-hrnet" \
    2>&1 | tee log/log.log