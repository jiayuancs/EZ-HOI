# export PYTHONPATH=$PYTHONPATH:"/workspace/code/OpenSourceModel/EZ-HOI"

CUDA_VISIBLE_DEVICES=2,3,4,5  python main_tip_finetune.py --world-size 4 \
 --pretrained "checkpoints/detr-r50-hicodet.pth"  \
 --output-dir checkpoints/vitb-default \
 --epochs 12  --use_insadapter  --num_classes 117 --use_multi_hot \
 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
 --clip_dir_vit checkpoints/pretrained_CLIP/ViT-B-16.pt \
 --batch-size 4  --logits_type "HO"  --port 1231 \
 --txtcls_pt   --img_align  --unseen_pt_inj  --img_clip_pt \
 --clip_img_file   hicodet_pkl_files/clipbase_img_hicodet_train  \


