# Object detection with a Vision Language Model

Zero-shot object detection on the course's own dataset
([`../02-Pytorch and CV/datasets/retail_products`](../02-Pytorch%20and%20CV/datasets/retail_products)),
with no training at all: a **multimodal LLM** running locally through **Ollama** reads
the image and writes the bounding boxes and classes in its answer.

| File | What it is |
| --- | --- |
| [`vlm_object_detection_ollama.ipynb`](vlm_object_detection_ollama.ipynb) | the full walkthrough — run this |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzo-stacchio/Deep-Learning-and-Computer-Vision-for-Business/blob/main/03-MLLM/vlm_object_detection_ollama.ipynb)

## From a trained detector to a prompt

In [`02_detection/yolov12_retail_products`](../02-Pytorch%20and%20CV/02_detection/yolov12_retail_products/)
the same task is solved by fine-tuning YOLOv12 on 294 annotated images. A **Vision
Language Model (VLM)** — a language model that also accepts images — needs no
training data: the classes are just words in the prompt.

| | YOLOv12 (module 02) | VLM (this folder) |
| --- | --- | --- |
| Training data | 294 annotated images | none (zero-shot) |
| Classes | fixed at training time | whatever the prompt says |
| Output | boxes + class scores | text (JSON) that we parse |
| Speed | milliseconds per image | seconds per image |

## The stack

* **[Ollama](https://ollama.com)** — downloads and runs open models locally (on top of
  llama.cpp) and serves them through an HTTP API on port `11434`, with a Python client.
* **[Qwen3.5 4B](https://ollama.com/library/qwen3.5)**, tag **`qwen3.5:4b-q4_K_M`** — a
  natively multimodal model from Alibaba. 4 billion parameters **quantized to 4 bits**
  (`q4_K_M`, the llama.cpp "K-quant, medium" scheme): **3.4 GB** of weights instead of
  ~9 GB in 16-bit, so it fits comfortably on a free Colab T4 (15 GB).

Other tags worth trying: `qwen3.5:4b-q8_0` (8-bit), `qwen3.5:9b` (bigger),
`qwen3-vl:4b-instruct-q4_K_M` (the previous, vision-specialised generation).

## How a language model returns boxes

The model only produces text. Qwen models were trained to answer grounding requests by
*writing* coordinates, e.g. `{"label": "aqua", "bbox_2d": [503, 354, 684, 547]}` with
`bbox_2d = [x1, y1, x2, y2]`.

> **Qwen3.5 writes coordinates on a normalized 0–1000 scale**, whatever the image size.
> Convert with `x_pixel = x / 1000 * width`, `y_pixel = y / 1000 * height`. Read as pixels,
> the boxes of our 512×512 images are completely wrong. This convention is not stated on
> the model page; we measured it against the ground truth.

The notebook makes the answer machine-readable in three steps:

1. **Structured outputs** — a Pydantic schema passed as `format=` constrains generation:
   the answer is always valid JSON, `label` must be one of the 6 classes (`Literal`) and
   `bbox_2d` has exactly 4 integers. The free-form alternative (JSON inside markdown,
   missing fields, invented labels) is shown first for comparison.
2. **`think=False`** — Qwen3.5 reasons before answering by default; for perception
   tasks this is much slower for no clear gain.
3. **`temperature=0`** — deterministic answers, so results are reproducible.

## Notebook outline

1. Install Ollama (plus `zstd`, required by the installer and missing in Colab), start
   `ollama serve` as a background process, pull the model.
2. Download a `mix` test image **and its Pascal VOC annotation** from this repository.
3. Chat with the model about the image; check with `ollama ps` that it runs on the GPU.
4. Detection prompt: free-form answer vs structured output.
5. 0–1000 → pixels, predictions drawn over the ground truth.
6. Evaluation: IoU, one-to-one matching per class (Hungarian algorithm), precision and
   recall at IoU ≥ 0.5, on all **26 `mix` test images** (each contains the 6 products).
7. Open-vocabulary detection: labels that are not in the dataset.
8. Discussion and exercises.

## Results

Measured by running the notebook end to end with `qwen3.5:4b-q4_K_M` (Ollama 0.33,
NVIDIA RTX 4070 Laptop GPU), on the 26 `mix` test images = 156 objects:

| Precision | Recall | Mean IoU of correct boxes | Time |
| --- | --- | --- | --- |
| 94.5% | 77.6% | 0.82 | ~3 s / image |

| Class | aqua | chitato | indomie | pepsodent | shampoo | tissue |
| --- | --- | --- | --- | --- | --- | --- |
| Recall | 81% | 88% | 70% | 88% | 72% | 65% |

What the numbers say:

* **When the model outputs a box, it is almost always right** (few false positives) and
  well placed (IoU 0.82).
* **The errors are missed objects.** On a few cluttered, top-down shots the model stops
  after 2–3 packages. Rewording the prompt to insist on listing every package did not
  help (recall 75.6%, and slower).
* **Open vocabulary is flexible but fragile.** Without the class list, the same model
  may skip packages, repeat one, use vague labels ("product package") or invent brand
  names: the closed list in the prompt and in the schema is what keeps it focused.
* A detector fine-tuned on the domain, like the YOLOv12 of module 02, is more complete
  and hundreds of times faster. The VLM's advantage is that it needs no labels and no
  training, and that its vocabulary can change with the prompt.

Times on a Colab T4 are of the same order (a few seconds per image); the first request
is slower because it loads the model into GPU memory.

## Running it locally

```bash
# 1. install Ollama: https://ollama.com/download (Windows/macOS app, or the Linux script)
ollama pull qwen3.5:4b-q4_K_M      # the Ollama server must be running (app or `ollama serve`)

# 2. Python dependencies
pip install ollama pydantic requests matplotlib pandas scipy pillow
```

Then open the notebook and skip the installation cells: the server-start cell detects
an already running Ollama and does nothing. A GPU with ~4 GB of free memory is enough;
on CPU the model works, but expect tens of seconds per image.

## Troubleshooting

* **`This version requires zstd for extraction`** — run `apt-get install -y zstd` before
  the install script (the notebook does it).
* **The server does not start** — read `ollama.log`, written by the notebook next to
  itself.
* **`ollama ps` shows `CPU` instead of `100% GPU`** — check that the Colab runtime has a
  GPU; Ollama needs an NVIDIA driver ≥ 550.
* **Boxes in the wrong place** — check the coordinate convention: Qwen3.5 and Qwen3-VL
  use 0–1000, other VLMs use pixels or 0–1.
