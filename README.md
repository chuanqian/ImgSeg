# ImgSeg

## Major Features

- Unified Benchmark

  we provide a unified training script for various segmentation methods.

- Simple and Effective

  Thanks to **pytorch lightning** and **timm** , the code is easy for further development.

- Supported Remote Sensing Datasets

    - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
    - [UAVid](https://uavid.nl/)
    - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)
    - [OpenEarthMap](https://open-earth-map.org/)
    - More datasets will be supported in the future.

- Multi-scale Training and Testing
- Inference on Huge Remote Sensing Images

## Folder Structure

Prepare the following folders to organize this repo:

```none
airs
├── ImgSeg (code)
├── pretrain_weights (pretrained weights of backbones, such as vit, swin, etc)
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── train_val (processed)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:

```
conda create -n airs python=3.8
conda activate airs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r ImgSeg/requirements.txt
```

## Pretrained Weights of Backbones

[Baidu Disk](https://pan.baidu.com/s/1foJkxeUZwVi5SnKNpn6hfg) : 1234

[Google Drive](https://drive.google.com/drive/folders/1ELpFKONJZbXmwB5WCXG7w42eHtrXzyPn?usp=sharing)

## Data Preprocessing

Download the datasets from the official website and split them yourself.

**Vaihingen**

Generate the training set.

```
python ImgSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/train_images" \
--mask-dir "data/vaihingen/train_masks" \
--output-img-dir "data/vaihingen/train/images_1024" \
--output-mask-dir "data/vaihingen/train/masks_1024" \
--mode "train" --split-size 1024 --stride 512 
```

Generate the testing set.

```
python ImgSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks_eroded" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded
```

Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.

```
python ImgSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt
```

As for the validation set, you can select some images from the training set to build it.

**Potsdam**

```
python ImgSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/train_images" \
--mask-dir "data/potsdam/train_masks" \
--output-img-dir "data/potsdam/train/images_1024" \
--output-mask-dir "data/potsdam/train/masks_1024" \
--mode "train" --split-size 1024 --stride 1024 --rgb-image 
```

```
python ImgSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks_eroded" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded --rgb-image
```

```
python ImgSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt --rgb-image
```

**UAVid**

```
python ImgSeg/tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train_val" \
--output-img-dir "data/uavid/train_val/images" \
--output-mask-dir "data/uavid/train_val/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python ImgSeg/tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train" \
--output-img-dir "data/uavid/train/images" \
--output-mask-dir "data/uavid/train/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python ImgSeg/tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_val" \
--output-img-dir "data/uavid/val/images" \
--output-mask-dir "data/uavid/val/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

**LoveDA**

```
python ImgSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python ImgSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python ImgSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python ImgSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```

## Training

"-c" means the path of the config, use different **config** to train different models.

```
python ImgSeg/train_supervision.py -c ImgSeg/config/uavid/unetformer.py
python ImgSeg/train_supervision.py -c ImgSeg/config/potsdam/unetformer.py
python ImgSeg/train_supervision.py -c ImgSeg/config/vaihingen/hdpa.py
python ImgSeg/train_supervision.py -c ImgSeg/config/vaihingen/backbone_dfsi_mdaf.py
python ImgSeg/train_supervision.py -c ImgSeg/config/potsdam/backbone_dfsi_mdaf.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models.

"-o" denotes the output path

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format

**Vaihingen**

```
python ImgSeg/vaihingen_test.py -c ImgSeg/config/vaihingen/dcswin.py -o fig_results/vaihingen/dcswin --rgb -t 'd4'
python ImgSeg/vaihingen_test.py -c ImgSeg/config/vaihingen/unetformer.py -o fig_results/vaihingen/unetformer --rgb -t 'd4'
python ImgSeg/vaihingen_test.py -c ImgSeg/config/vaihingen/backbone_dfsi_mdaf.py -o fig_results/vaihingen/backbone_dfsi_mdaf --rgb -t 'd4'
```

**Potsdam**

```
python ImgSeg/potsdam_test.py -c ImgSeg/config/potsdam/dcswin.py -o fig_results/potsdam/dcswin --rgb -t 'lr'
python ImgSeg/potsdam_test.py -c ImgSeg/config/potsdam/unetformer.py -o fig_results/potsdam/unetformer --rgb -t 'lr'
python ImgSeg/potsdam_test.py -c ImgSeg/config/potsdam/hdpa.py -o fig_results/potsdam/hdpa --rgb -t 'lr'
python ImgSeg/potsdam_test.py -c ImgSeg/config/potsdam/backbone_dfsi_mdaf.py -o fig_results/potsdam/backbone_dfsi_mdaf --rgb -t 'lr'
```

**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))

```
python ImgSeg/loveda_test.py -c ImgSeg/config/loveda/dcswin.py -o fig_results/loveda/dcswin_test -t 'd4'
```

**UAVid** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/7302))

```
python ImgSeg/inference_uavid.py \
-i 'data/uavid/uavid_test' \
-c ImgSeg/config/uavid/unetformer.py \
-o fig_results/uavid/unetformer_r18 \
-t 'lr' -ph 1152 -pw 1024 -b 2 -d "uavid"
```

## Inference on huge remote sensing image

```
python ImgSeg/inference_huge_image.py \
-i data/vaihingen/test_images \
-c ImgSeg/config/vaihingen/dcswin.py \
-o fig_results/vaihingen/dcswin_huge \
-t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```