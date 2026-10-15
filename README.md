# 🐱 Cat Detection — Selective Search + MobileNetV2 (R-CNN style)

Object detection mini-project that finds **cats in an image** and draws bounding boxes around them.
It follows the classic R-CNN idea: propose candidate regions, classify each region with a CNN, then keep the best boxes.

> Course project — [Course name / code], [University / Department]

## Group members

| Name | Roll No. |
|------|----------|
| [Name 1] | [Roll no.] |
| [Name 2] | [Roll no.] |

## 1. How it works

1. **Ground truth** – every `cat` bounding box is read from the Pascal-VOC style `.xml` annotation files.
2. **Region proposals** – OpenCV *Selective Search* (fast mode) generates candidate boxes for every image.
3. **Training crops** – each proposal is compared with the ground-truth boxes using IoU:
   - IoU > 0.5 → class `cat`
   - IoU < 0.1 with *all* boxes → class `no_category` (max. 50 per image, min. 20×20 px)
   - crops are saved as RGB and resized to 224×224.
4. **Split by image** – images (not crops) are split 70 % train / 15 % validation / 15 % test, so crops of the same photo never appear in both training and test.
5. **Classifier** – MobileNetV2 (ImageNet weights, frozen) with `AveragePooling2D → Dense(128) → Dropout(0.3) → Dense(2, softmax)`.
   Random crop + horizontal flip are part of the model (active only in training); inputs are scaled to [-1, 1] as MobileNetV2 expects. `class_weight = {cat: 5, no_category: 1}` compensates for the many background crops. Early stopping on validation loss.
6. **Detection** – Selective Search runs on a new image (resized so the longest side is ≤ 640 px), every proposal is classified, and **Non-Maximum Suppression** (IoU 0.3, cat probability ≥ 0.9, max. 5 boxes) keeps the best boxes. If no region reaches the threshold, nothing is drawn.

## 2. Repository structure

```
.
├── CatDetectionModel.ipynb   # Full pipeline: data prep, training, evaluation, detection
├── sample_input.jpg          # Example image for the detection demo
├── requirements.txt
├── .gitignore
└── README.md
```

Created when you run the notebook (optional to commit): `cat_detector.keras` (trained model) and `results/detection_result.jpg` (detection output).

Not stored in the repo (too large / generated): the dataset in `archive/` and the `training/` crops.

## 3. Dataset

**Cats and Dogs Classification Dataset** by Bhavik Jikadara on Kaggle:
https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

Please check the dataset page for its license and terms of use.

1. Download the dataset from the link above and extract it.
2. In the first code cell of the notebook, set `DATA_DIR` to the path of the folder that contains the cat images and their annotations (`cat.N.jpg` + `cat.N.xml`), for example:
   ```python
   DATA_DIR = Path("archive")
   ```
   Sub-folders are searched automatically.

By default only images smaller than 100 KB are used and the notebook processes at most 350 images (`MAX_IMAGES`), because Selective Search takes about 5 s per image on a CPU. Set `MAX_IMAGES = None` to use everything.

## 4. Setup & run

```bash
git clone https://github.com/Wajeehaali014/CatDetectionModel.git
cd CatDetectionModel

python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux / macOS

pip install -r requirements.txt
jupyter notebook CatDetectionModel.ipynb
```

Run the cells from top to bottom. Notes:

- `opencv-contrib-python` is required (Selective Search lives in `cv2.ximgproc`). Do **not** install plain `opencv-python` alongside it.
- Generating the crops is skipped if the `training/` folder already exists (set `REGENERATE_CROPS = True` to rebuild).
- To only run the detection demo, put `cat_detector.keras` and your `sample_input.jpg` next to the notebook and run the imports, the helper functions and the last section. The model is loaded from disk automatically.

## 5. Results

Results have not been added yet. After running the notebook, section 7 prints precision and recall on the held-out test images, and section 8 saves the detection output to `results/detection_result.jpg`.

<!-- Add after running:
| Metric (held-out test images) | cat | no_category |
|-------------------------------|-----|-------------|
| Precision | | |
| Recall | | |

![Detection result](results/detection_result.jpg)
-->

## 6. Known limitations

- Trained on a small subset (≤ 350 images). Test numbers come from crops of a few dozen held-out images and can vary a lot with the random split.
- Crop-level precision/recall is reported, not detection mAP at IoU 0.5, which is the standard object-detection metric.
- The detector can miss cats that Selective Search does not propose a good box for, and boxes are only as tight as the proposals.
- Slow compared with modern detectors (YOLO, SSD, Faster R-CNN): thousands of proposals are classified per image.
- Only the `cat` class is detected.

## 7. Credits

- R-CNN approach inspired by [PyImageSearch: R-CNN object detection with Keras, TensorFlow and Deep Learning](https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/).
- IoU function adapted from a Stack Overflow answer (linked in the notebook).
- Dataset: see Section 3.
