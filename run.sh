# preprocess training dataset to hdf5 file
# python make_style_dataset.py --train_dir data/mscoco/train2014/ --val_dir data/mscoco/val2014/ --output_file data/ms-coco-256.h5

# preprocess gram matrices for style image
# python make_gram_dataset.py --gram_dataset_path data/bf_grams.h5 --style_dir images/style-images/ \
# --style_imgs bird_flower.jpg --style_img_size 384 --gpu 0

# train with defaults
python train.py --norm_by_channels --coco_path data/ms-coco-256.h5 \
 --gram_dataset_path data/bf_grams.h5 --checkpoint_path results/bf.h5

# train with pytorch settings
python train.py --lr 1e-3 --content_weight 1e5 --style_weight 1e10 --tv_weight 1 \
--norm_by_channels --coco_path data/ms-coco-256.h5 \
--gram_dataset_path data/bf_grams.h5 --checkpoint_path results/bf_pytorch.h5

# evaluate
python fast_style_transfer.py --checkpoint_path results/bf.h5 --input_path images/content-images\
--output_path results/ --use_style_name
