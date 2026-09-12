# YOLO Object Detection Training on Custom Dataset

This notebook trains a YOLO (You Only Look Once) object detection model on a **custom dataset**.  
The main steps are:

1. **Install and Import Dependencies**
   - Installs required libraries (e.g., `ultralytics`, `torch`, etc.).
   - Imports necessary Python packages.

2. **Prepare Dataset**
   - Organizes the dataset into YOLO format:
     ```
     dataset/
     ├── train/
     │   ├── images/
     │   └── labels/
     ├── val/
     │   ├── images/
     │   └── labels/
     └── data.yaml
     ```
   - The `data.yaml` file contains dataset configuration (class names, number of classes, train/val paths).

3. **Load Pre-trained YOLO Model**
   - Uses a YOLO model (e.g., `yolov8n.pt`) as a starting point.
   - Fine-tunes it on the custom dataset.

4. **Train the Model**
   - Runs the training loop with parameters like:
     - `epochs`: number of training iterations.
     - `batch`: batch size.
     - `imgsz`: image size.
     - `device`: CPU or GPU.
   - Tracks loss and accuracy metrics.

5. **Evaluate the Model**
   - Tests model performance on the validation dataset.
   - Reports metrics like **mAP (mean Average Precision)**, precision, recall, etc.

6. **Visualize Results**
   - Shows training curves (loss, precision, recall).
   - Displays sample predictions on images.

7. **Save and Export Model**
   - Saves the trained weights (e.g., `best.pt`).
   - Can be used for inference on new images.

---

### Summary
This script fine-tunes a YOLO object detection model on your own dataset by:
- Preparing data in YOLO format,  
- Training the model,  
- Evaluating performance, and  
- Saving the trained model for future predictions.
