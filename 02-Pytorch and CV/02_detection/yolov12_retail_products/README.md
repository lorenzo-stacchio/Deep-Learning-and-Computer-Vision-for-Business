# YOLOv12 on the `retail_products` dataset

End-to-end object detection on the course's own dataset
([`../../datasets/retail_products`](../../datasets/retail_products)): inference with
the COCO-pretrained model, annotation conversion, fine-tuning, evaluation, and
deployment export.

| File | What it is |
| --- | --- |
| [`yolov12_retail_products.ipynb`](yolov12_retail_products.ipynb) | the full walkthrough — run this |
| [`voc_to_yolo.py`](voc_to_yolo.py) | the Pascal VOC → YOLO conversion, as a standalone script |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzo-stacchio/Deep-Learning-and-Computer-Vision-for-Business/blob/main/02-Pytorch%20and%20CV/02_detection/yolov12_retail_products/yolov12_retail_products.ipynb)

## Why YOLOv12

Every YOLO up to v11 is essentially convolutional. Attention has more modelling
capacity, but vanilla self-attention is quadratic in the number of pixels and has
poor memory locality, so it kept losing to CNNs on the real-time speed/accuracy
trade-off.

YOLOv12 ([Tian et al., 2025](https://arxiv.org/abs/2502.12524)) makes attention
competitive there, through two changes worth remembering:

* **Area Attention (A²)** — split the feature map into a few large regions and
  attend within each. Keeps a wide receptive field at a fraction of the cost.
* **R-ELAN** — a residual ELAN variant with scaled residuals, which is what makes
  the larger attention models train stably.

FlashAttention is **optional**: Ultralytics falls back to a standard implementation
on GPUs that lack it, only slower.

> YOLOv12 is a community-driven release — Ultralytics flags possible training
> instability and elevated memory use, and recommends YOLO11/YOLO26 for production.
> It is used here because it is the clearest example of attention entering the
> real-time detection family. The pipeline is identical for any other checkpoint:
> change `MODEL_NAME`.

## The dataset

380 images at 512×512, annotated in Pascal VOC XML, of 6 Indonesian supermarket
products: `aqua` (water), `chitato` (crisps), `indomie` (instant noodles),
`pepsodent` (toothpaste), `shampoo`, `tissue`.

| split | images | composition |
| --- | --- | --- |
| `train` | 294 | 237 single-product shots + 57 `mix_*` shots containing all 6 |
| `test` | 86 | same composition; used as the validation split |

Classes are balanced (95–97 boxes each in train), objects are large relative to the
frame, and nothing is occluded — an easy detection problem, deliberately, so that
the pipeline rather than the difficulty is the subject.

## Converting Pascal VOC to YOLO

Ultralytics cannot read Pascal VOC. It wants one `.txt` per image with
`<class_id> <x_center> <y_center> <width> <height>`, all **normalised to [0, 1]**:

```
retail_products/                          retail_products_yolo/
├── images/{train,test}/aqua (1).jpg   →   ├── train/{images,labels}/aqua_1.{jpg,txt}
└── annotations/{train,test}/aqua (1).xml  ├── val/{images,labels}/...
                                           └── data.yaml
```

Two properties of this dataset make a naive converter silently wrong, and both are
handled in `voc_to_yolo.py`:

1. **`<filename>` does not match the file on disk.** The XML for `aqua (1).xml`
   says `20210518_201521` — the name of the original shot. Images and annotations
   must be paired on the *stem of the XML file*.
2. **`<path>` is the annotator's own Windows path**, and is unusable.

Source filenames are also normalised (`aqua (1).jpg` → `aqua_1.jpg`) to avoid
spaces and parentheses in paths and CLI commands.

Run it standalone:

```bash
python voc_to_yolo.py                                  # uses the repo's dataset
python voc_to_yolo.py --src /path/to/voc --dst /path/to/yolo
```

The notebook contains the same code inline and verifies the result two ways: the
per-class object counts must match the VOC side exactly, and every normalised
coordinate must fall inside [0, 1]. Both are `assert`s — a conversion bug fails
loudly instead of quietly costing you 20 points of mAP.

## Fine-tuning notes

With 294 training images the settings that matter are:

* **start from COCO weights.** Training YOLOv12 from a `.yaml` config on 294 images
  does not converge to anything useful.
* **`mosaic=1.0`.** Stitching 4 images together is what lets a dataset of mostly
  single-object photographs learn crowded scenes. `close_mosaic=10` disables it for
  the final epochs so the model ends on realistic images.
* **keep `hsv_h` low.** Colour is a large part of what distinguishes these products;
  heavy hue jitter destroys the signal. Brightness (`hsv_v`) is the opposite — shops
  have wildly different lighting, so vary it.
* **`patience=25`.** A small dataset needs many epochs but overfits if left running.

## Caveat on the reported metrics

The notebook uses the 86 test images as the validation set, so the reported mAP is
measured on data that also drove early stopping and best-checkpoint selection.
That is a deliberate simplification for a dataset this small, and it means the
numbers are mildly optimistic. Exercise 1 in the notebook fixes it with a proper
three-way split.
