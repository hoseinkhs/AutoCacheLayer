
    # --distillation_test \
#${XD}_names_img_list_test.txt"
mkdir "log"
XD=127
BS=32
backbone="MobileFaceNet"
classifier_type="Dense2Layer"
exit_type="Dense2LayerTemp"
exit_epoch=19
trial="attention"
# --run_server \
python train.py \
    --experiment "Face" \
    --shrink \
    --num_classes ${XD} \
    --fine_tune 0 \
    --trial "${trial}" \
    --train_epochs 20 \
    --online_train_epochs 20 \
    --train_device "cuda:0" \
    --test_device "cuda:0" \
    --exit_model_path "./out_dir/${backbone}/exits2/${exit_type}/" \
    --data_root "./data/test/lfw/images_cropped" \
    --train_file "./data/test/lfw/${XD}_names_img_list_train.txt" \
    --test_file "./data/test/lfw/${XD}_names_img_list_test.txt" \
    --classes_file "./data/test/lfw/${XD}_names.txt" \
    --backbone_type "${backbone}" \
    --backbone_conf_file "backbone_conf.yaml" \
    --backbone_model_path "./models/${backbone}-Epoch_17.pt" \
    --classifier_type "${classifier_type}" \
    --classifier_conf_file "classifier_conf.yaml" \
    --classifier_model_path "training_mode/classifier_training/out_dir/${backbone}/${XD}id/${classifier_type}/Epoch_19.pt" \
    --exit_type "${exit_type}" \
    --exit_conf_file "./exit_conf.yaml" \
    --lr 0.1 \
    --out_dir "out_dir/${backbone}" \
    --step "10, 13, 16" \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size $BS \
    --test_batch_size 100 \
    --momentum 0.9 \
    --log_dir "log" \
    --tensorboardx_logdir "mv-hrnet" \
    2>&1 | tee log/log.log
