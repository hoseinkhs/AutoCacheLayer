mkdir 'log'
XD=127
python train.py \
    --train_device 'cuda:0' \
    --test_device 'cpu' \
    --resume \
    --exit_model_paths './out_dir/Exit_0_epoch_19.pt' './out_dir/Exit_1_epoch_19.pt' './out_dir/Exit_2_epoch_19.pt' \
    --data_root '../../data/test/lfw/images_cropped' \
    --train_file "../../data/test/lfw/${XD}_names_img_list_train.txt" \
    --test_file "../../data/test/lfw/${XD}_names_img_list_test.txt" \
    --names_file "../../data/test/lfw/${XD}_names.txt" \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --backbone_model_path '../../models/MobileFaceNet-Epoch_17.pt' \
    --classifier_type 'Dense2Layer' \
    --classifier_conf_file '../classifier_conf.yaml' \
    --classifier_model_path "../classifier_training/out_dir/${XD}_logsoftmax/Epoch_19.pt" \
    --num_exits 3 \
    --exit_type 'Dense2Layer' \
    --exit_conf_file '../exit_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir' \
    --epoches 0 \
    --previous_epoch_num 19 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 32 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-hrnet' \
    2>&1 | tee log/log.log


