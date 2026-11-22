# Dataset

184 facial acne images in three classes, manually cropped to 150 x 150 pixels:

```
compiled/
├── nodules/    51 images
├── papule/     61 images
└── pustules/   72 images
```

The images were collected from public sources, including Skin90, DermNet, and
other online dermatology repositories. They remain the property of their original
owners and are subject to those sources' terms. They are included here for
research and reproducibility only. Check each source's terms before reusing
them, particularly for commercial purposes.

`seg_fcm.py` and `seg_kmeans.py` read this folder, create a stratified 70/30
train/test split in `split_dataset/`, and write segmentation output to `seg_fcm/`
and `seg_results_test_gray/`. Those three generated folders are git-ignored.
