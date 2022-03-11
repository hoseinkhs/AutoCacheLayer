    # --shrink \    
   
    # --distillation_test \
#${XD}_names_img_list_test.txt"
mkdir "log"
XD=365
BS=32
backbone="PlacesAlexNet" #"PlacesResnet"

exit_type="ConvDense"
exit_epoch=0
trial="accuracies-no-shrink"

python train.py \
    --experiment "Place" \
    --num_classes ${XD} \
    --fine_tune 0 \
    --trial "${trial}" \
    --train_epochs 0 \
    --train_device "cuda:0" \
    --test_device "cuda:0" \
    --exit_model_path "./out_dir/${backbone}/exits/${exit_type}/" \
    --data_root "../../data/places/places365_standard" \
    --train_file "../../data/places/places365_standard/val.txt" \
    --test_file "../../data/places/places365_standard/val.txt" \
    --classes_file "../../data/places/places365_standard/names.txt" \
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