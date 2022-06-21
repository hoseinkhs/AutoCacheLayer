mkdir "log"
XD=10
BS=16
backbone="Resnet18"
exit_type="Dense2LayerTemp"
exit_epoch=19
trial="40-mem-profile"
experiment="Cifar10"
# --run_server \
# --search_cache_models \
# --count_flops \
python train.py \
    --experiment "${experiment}" \
    --shrink \
    --num_classes ${XD} \
    --fine_tune 0 \
    --trial "${trial}" \
    --train_epochs 0 \
    --train_device "cuda:0" \
    --test_device "cpu" \
    --exit_model_path "./out_dir/${experiment}/${backbone}/exits/${exit_type}/" \
    --data_root "./data/cifar10" \
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
    --test_batch_size 32 \
    --momentum 0.9 \
    --log_dir "log" \
    --tensorboardx_logdir "mv-hrnet" \
    2>&1 | tee log/log.log