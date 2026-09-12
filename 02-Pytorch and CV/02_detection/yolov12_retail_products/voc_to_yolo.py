"""Convert the `retail_products` dataset from Pascal VOC to the Ultralytics YOLO format.

Source layout::

    retail_products/
    ├── images/{train,test}/aqua (1).jpg
    └── annotations/{train,test}/aqua (1).xml

Target layout::

    retail_products_yolo/
    ├── train/{images,labels}/aqua_1.{jpg,txt}
    ├── val/{images,labels}/...
    └── data.yaml

Two quirks of this dataset are worth knowing about:

* the ``<filename>`` tag of the XML files holds the name of the original shot
  (e.g. ``20210518_201521``) and **not** the name of the file on disk, so images
  and annotations must be paired on the *stem* of the XML file;
* the ``<path>`` tag points to a Windows folder on the annotator's machine and
  is therefore unusable.

Command line usage::

    python voc_to_yolo.py --src ../../datasets/retail_products --dst ../../datasets/retail_products_yolo
"""

from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# The 6 products in the dataset, in alphabetical order: the index in this list
# is the class id written into the YOLO .txt label files.
CLASSES = ["aqua", "chitato", "indomie", "pepsodent", "shampoo", "tissue"]

# Source files are named "aqua (1).jpg". Spaces and parentheses are legal but
# awkward in paths, CLI commands and logs, so we normalise them to "aqua_1".
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_stem(stem: str) -> str:
    """`aqua (1)` -> `aqua_1`."""
    return _SAFE.sub("_", stem).strip("_")


def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Read a VOC XML file and return (width, height, objects).

    Each object is ``(class_name, xmin, ymin, xmax, ymax)`` in pixels.
    """
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    width = int(float(size.find("width").text))
    height = int(float(size.find("height").text))

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        objects.append((name, xmin, ymin, xmax, ymax))

    return width, height, objects


def voc_box_to_yolo(box, width: int, height: int) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) in pixels -> (xc, yc, w, h) normalised to [0, 1]."""
    xmin, ymin, xmax, ymax = box

    # A few annotations can overshoot the image border by a pixel or two.
    xmin, xmax = max(0.0, min(xmin, xmax)), min(float(width), max(xmin, xmax))
    ymin, ymax = max(0.0, min(ymin, ymax)), min(float(height), max(ymin, ymax))

    xc = ((xmin + xmax) / 2) / width
    yc = ((ymin + ymax) / 2) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return xc, yc, w, h


def convert_split(src: Path, dst: Path, src_split: str, dst_split: str) -> Counter:
    """Convert one split (`train`/`test` of the source) into a YOLO split."""
    img_src = src / "images" / src_split
    ann_src = src / "annotations" / src_split
    img_dst = dst / dst_split / "images"
    lbl_dst = dst / dst_split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    for xml_path in sorted(ann_src.glob("*.xml")):
        # Pair on the file stem, not on the <filename> tag.
        image_path = next((p for p in img_src.glob(xml_path.stem + ".*")), None)
        if image_path is None:
            stats["missing_images"] += 1
            continue

        width, height, objects = parse_voc(xml_path)

        lines = []
        for name, *box in objects:
            if name not in CLASSES:
                stats[f"unknown_class:{name}"] += 1
                continue
            xc, yc, w, h = voc_box_to_yolo(box, width, height)
            if w <= 0 or h <= 0:
                stats["degenerate_boxes"] += 1
                continue
            lines.append(f"{CLASSES.index(name)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            stats[name] += 1

        stem = safe_stem(xml_path.stem)
        shutil.copy2(image_path, img_dst / f"{stem}{image_path.suffix.lower()}")
        (lbl_dst / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["images"] += 1

    return stats


def write_data_yaml(dst: Path) -> Path:
    """Write the data.yaml Ultralytics reads (absolute path, to avoid ambiguity)."""
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    yaml_path = dst / "data.yaml"
    yaml_path.write_text(
        f"path: {dst.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: val/images\n"
        "\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def convert(src: Path, dst: Path) -> Path:
    """Convert the whole dataset. The source `test` split becomes `val`."""
    if dst.exists():
        shutil.rmtree(dst)

    for src_split, dst_split in (("train", "train"), ("test", "val")):
        stats = convert_split(src, dst, src_split, dst_split)
        print(f"[{dst_split}] {stats['images']} images")
        for cls in CLASSES:
            print(f"    {cls:<12} {stats[cls]} boxes")
        issues = {k: v for k, v in stats.items() if k not in CLASSES and k != "images"}
        if issues:
            print(f"    issues: {issues}")

    return write_data_yaml(dst)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=here / ".." / ".." / "datasets" / "retail_products")
    parser.add_argument("--dst", type=Path, default=here / ".." / ".." / "datasets" / "retail_products_yolo")
    args = parser.parse_args()

    yaml_path = convert(args.src.resolve(), args.dst.resolve())
    print(f"\ndata.yaml written to: {yaml_path}")
