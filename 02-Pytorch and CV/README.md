# PyTorch and Computer Vision

This module is organised by **computer vision task**. Each folder is a self-contained
step of the same story: first recognise *what* is in an image, then *where* it is,
then *where it goes*, then *which pixels* belong to it.

| # | Task | Question it answers | Output |
| --- | --- | --- | --- |
| [01](01_classification/) | Classification | *What* is in the image? | one label per image |
| [02](02_detection/) | Object detection | *What* and *where*? | boxes + labels |
| [03](03_tracking/) | Tracking | *Where does it go* across frames? | boxes + labels + persistent IDs |
| [04](04_segmentation/) | Segmentation | *Which pixels* belong to it? | masks |
| [05](05_pose_estimation/) | Pose estimation | *How is the body arranged*? | keypoints |

---

## 01 — Classification

One label for the whole image: the foundation everything else is built on.

* [`basic_cnn/`](01_classification/basic_cnn/) — CNN fundamentals on FashionMNIST:
  convolutions, pooling, the full PyTorch training loop written by hand.
* [`cnn_finetuning/`](01_classification/cnn_finetuning/) — transfer learning with
  pretrained `torchvision` backbones (ResNet, VGG, DenseNet): feature extraction
  vs full fine-tuning, and when each is the right call.

## 02 — Object detection

Localise *and* classify every object in the frame.

* [`yolo_custom_dataset/`](02_detection/yolo_custom_dataset/) — the YOLO family with
  Ultralytics: CLI and Python SDK, COCO inference, fine-tuning on a custom dataset.
* [`yolov12_retail_products/`](02_detection/yolov12_retail_products/) — **end-to-end
  worked example**: the course's own `retail_products` dataset, converted from
  Pascal VOC and fine-tuned with the attention-based YOLOv12.
* [`mmdetection/`](02_detection/mmdetection/) — the same task through MMDetection,
  a config-driven framework: useful for comparing ecosystems.

## 03 — Tracking

Detection tells you there is a person in frame 1 and a person in frame 2.
Tracking tells you it is the *same* person — which is what turns detections into
business metrics.

* [`2_object_tracking.ipynb`](03_tracking/2_object_tracking.ipynb) — tracking
  fundamentals and the main algorithm families.
* [`ultralytics_yolo.ipynb`](03_tracking/ultralytics_yolo.ipynb) — `model.track()` on
  a real shopping-mall video, then retail analytics on the resulting tracks:
  footfall counting, spatial heatmaps, trajectory clustering.
* [`media/`](03_tracking/media/) — source video and rendered outputs.

## 04 — Segmentation

Pixel-level masks instead of rectangles.

* [`grounding_dino_sam/`](04_segmentation/grounding_dino_sam/) — zero-shot annotation
  with Grounding DINO (text prompt → boxes) and SAM (boxes → masks). The practical
  point: bootstrap a labelled dataset in minutes and spend your time verifying
  rather than drawing polygons.

## 05 — Pose estimation

Keypoints instead of boxes.

* [`AlphaPoseDemo.ipynb`](05_pose_estimation/AlphaPoseDemo.ipynb) — human pose
  estimation with AlphaPose.

---

## Shared resources

* [`datasets/retail_products/`](datasets/retail_products/) — 380 images (512×512) of
  6 supermarket products annotated in Pascal VOC format, split 294 train / 86 test.
  Classes: `aqua`, `chitato`, `indomie`, `pepsodent`, `shampoo`, `tissue`.
  Used by the detection notebooks.
* [`assets/`](assets/) — figures shared across notebooks.

## Running the notebooks

Every notebook runs on **Google Colab** (use the badge at the top of the notebook,
and enable a GPU via `Runtime` → `Change runtime type`) and locally in the cloned
repo. The notebooks that need the `retail_products` dataset locate it
automatically, whichever of the two you are on.
