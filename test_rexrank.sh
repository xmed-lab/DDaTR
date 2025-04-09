CUDA_VISIBLE_DEVICES=0, python main_test.py \
--n_gpu 1 \
--image_dir ./data/mimic_cxr/mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files/ \
--ann_path ./data/mimic_cxr/longitudinal_mimic_ddatr.json \
--dataset_name mimic_cxr \
--gen_max_len 150 \
--gen_min_len 100 \
--batch_size 16 \
--save_dir ./results/rexrank \
--seed 456789 \
--beam_size 3 \
--load_pretrained ./checkpoints/ddatr/model_best.pth

