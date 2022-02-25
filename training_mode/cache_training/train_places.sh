

    # --distillation_test \
#${XD}_names_img_list_test.txt"
mkdir "log"
XD=365
BS=2
backbone="PlacesResnet"
classifier_type="Dense2Layer"
exit_type="ConvDense"
exit_epoch=0
trial="008-places"

python train_places.py \
    --shrink \
    --fine_tune 0 \
    --trial "${trial}" \
    --train_epochs 0 \
    --train_device "cuda:0" \
    --test_device "cuda:0" \
    --data_root "../../data/places/places365_standard" \
    --train_file "../../data/places/places365_standard/val.txt" \
    --test_file "../../data/places/places365_standard/val.txt" \
    --names_file "../../data/places/places365_standard/names.txt" \
    --exit_model_paths "./out_dir/${backbone}/exits/${exit_type}/Exit_0.pt" \
                "./out_dir/${backbone}/exits/${exit_type}/Exit_1.pt" \
                "./out_dir/${backbone}/exits/${exit_type}/Exit_2.pt" \
    --backbone_conf_file "../backbone_conf.yaml" \
    --backbone_type "${backbone}" \
    --exit_type "${exit_type}" \
    --exit_conf_file "./exit_conf.yaml" \
    --lr 0.1 \
    --out_dir "out_dir/${backbone}" \
    --step "10, 13, 16" \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size $BS \
    --momentum 0.9 \
    --log_dir "log" \
    --tensorboardx_logdir "mv-hrnet" \
    2>&1 | tee log/log.log