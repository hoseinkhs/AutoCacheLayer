mkdir 'log'
python train.py \
    --data_root '../../data/test/lfw/images_cropped' \
    --train_file '../../data/test/lfw/15_names_img_list_train.txt' \
    --test_file '../../data/test/lfw/15_names_img_list_test.txt' \
    --names_file '../../data/test/lfw/15_names.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --backbone_model_path '../../models/MobileFaceNet-Epoch_17.pt' \
    --classifier_type 'Dense2Layer' \
    --classifier_conf_file '../classifier_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir/15_logsoftmax/' \
    --epoches 20 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 32 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-hrnet' \
    2>&1 | tee log/log.log


    # --resume \
    # --pretrain_model './out_dir/Epoch_17.pt' \