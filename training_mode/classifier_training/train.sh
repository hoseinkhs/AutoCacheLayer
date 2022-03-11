mkdir "log"
XD=127
backbone="MobileFaceNet"
classifier_type="Dense2Layer"
python train.py \
    --data_root "../../data/test/lfw/images_cropped" \
    --train_file "../../data/test/lfw/${XD}_names_img_list_train.txt" \
    --test_file "../../data/test/lfw/${XD}_names_img_list_test.txt" \
    --classes_file "../../data/test/lfw/${XD}_names.txt" \
    --backbone_type "${backbone}" \
    --backbone_conf_file "../backbone_conf.yaml" \
    --backbone_model_path "../../models/${backbone}-Epoch_17.pt" \
    --classifier_type "${classifier_type}" \
    --classifier_conf_file "../classifier_conf.yaml" \
    --lr 0.1 \
    --out_dir "out_dir/${backbone}/${XD}id/${classifier_type}/" \
    --epoches 20 \
    --step "10, 13, 16" \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 32 \
    --momentum 0.9 \
    --log_dir "log" \
    --tensorboardx_logdir "mv-hrnet" \
    2>&1 | tee log/log.log


    # --resume \
    # --pretrain_model "./out_dir/Epoch_17.pt" \