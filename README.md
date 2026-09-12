# Deep Learning and Computer Vision for Business

Ph.D. Course in [ECOSTATDATA at Bicocca University](https://www.unimib.it/didattica/offerta-formativa/dottorato-ricerca/corsi-dottorato/economia-statistica-e-data-science-ecostat)

![Teaser Image](./docs/teaser.webp)

This PhD course is designed to provide an in-depth understanding of how deep learning and computer vision techniques can be applied to solve business challenges. The course focuses on key practical applications, including retail product recognition, human pose estimation, and advanced image processing. Students will gain hands-on experience with PyTorch, developing models that address real-world business problems.

## Course Structure

The course is divided into three modules:

### [00 — OpenCV](./00_OpenCV/)

Classical image processing: the operations that come before any neural network, and
that still solve a surprising share of real problems on their own.

### [01 — PyTorch](./01-Pytorch/)

The framework itself: tensors, autograd, modules, and the anatomy of a training loop.

### [02 — PyTorch and Computer Vision](./02-Pytorch%20and%20CV/)

The core of the course, organised **by computer vision task**. Each folder answers a
different question about the same image, and the answers build on each other.

| # | Task | Question | Contents |
| --- | --- | --- | --- |
| [01](./02-Pytorch%20and%20CV/01_classification/) | **Classification** | *What* is in the image? | CNN fundamentals on FashionMNIST; transfer learning with pretrained `torchvision` backbones |
| [02](./02-Pytorch%20and%20CV/02_detection/) | **Object detection** | *What* and *where*? | YOLO with Ultralytics; a full YOLOv12 walkthrough on the course's retail dataset; the same task via MMDetection |
| [03](./02-Pytorch%20and%20CV/03_tracking/) | **Tracking** | *Where does it go* across frames? | `model.track()` on shopping-mall footage, then retail analytics: footfall counting, spatial heatmaps, trajectory clustering |
| [04](./02-Pytorch%20and%20CV/04_segmentation/) | **Segmentation** | *Which pixels* belong to it? | zero-shot annotation with Grounding DINO + SAM |
| [05](./02-Pytorch%20and%20CV/05_pose_estimation/) | **Pose estimation** | *How is the body arranged*? | human pose estimation with AlphaPose |

Shared across the module:

- **[`datasets/retail_products/`](./02-Pytorch%20and%20CV/datasets/retail_products/)** — 380 images (512×512) of 6 supermarket products annotated in Pascal VOC format, split 294 train / 86 test. Classes: `aqua`, `chitato`, `indomie`, `pepsodent`, `shampoo`, `tissue`. Used by the detection notebooks.
- **[`assets/`](./02-Pytorch%20and%20CV/assets/)** — figures shared across notebooks.

Every notebook runs on Google Colab (badge at the top of each notebook) as well as
locally in the cloned repo.

## Other Resources

- [YOLO Training Notebook](https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/master/Chapter08/Training_YOLO.ipynb)
- [Object Detection with DETR](https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/master/Chapter15/Object_detection_with_DETR.ipynb)
- [OpenPose Sign Language Model](https://colab.research.google.com/github/changsin/DL/blob/main/notebooks/openpose_sign_language.ipynb#scrollTo=FOdkDhb6ga6N)

Additional materials and datasets, including retail product datasets, can be accessed through:
- [Retail Product Dataset on Kaggle](https://www.kaggle.com/datasets/hafizyusufheraldi/retail-product-dataset?resource=download)
- [Retail Product Checkout Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset/data)

