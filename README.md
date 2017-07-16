# Moving sprites object detection

## Training
```bash
python train.py
```

```bash
tensorboard --logdir=models
```

## Detection
```bash
python detect.py \
    models/model_2017.07.13-01\:20\:37/ \
    ../imgs_detected \
    $(ls ../imgs/*)
```