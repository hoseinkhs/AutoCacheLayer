# mkdir 'log'
# python test.py \
#     --data_root '../../data/test/lfw/images_cropped' \
#     --train_file '../../data/test/lfw/127_names_img_list_train.txt' \
#     --test_file '../../data/test/lfw/127_names_img_list_test.txt' \
#     --names_file '../../data/test/lfw/127_names.txt' \
#     --backbone_type 'MobileFaceNet' \
#     --backbone_conf_file '../backbone_conf.yaml' \
#     --backbone_model_path '../../models/MobileFaceNet-Epoch_17.pt' \
#     --classifier_type 'Dense2Layer' \
#     --classifier_conf_file '../classifier_conf.yaml' \
#     --classifier_model_path '../classifier_training/out_dir/Epoch_17.pt' \
#     --exit_models_paths '../cache_training/out_dir/Exit_2_epoch_24.pt' \
#     --exit_models_paths '../cache_training/out_dir/Exit_2_epoch_24.pt' \
#     --exit_models_paths '../cache_training/out_dir/Exit_2_epoch_24.pt' \
#     --exit_type 'Dense2Layer' \
#     --exit_conf_file '../exit_conf.yaml' \
#     --batch_size 32 \
#     --out_dir 'out_dir' \
#     --log_dir 'log' \
#     --tensorboardx_logdir 'mv-hrnet' \
#     2>&1 | tee log/log.log