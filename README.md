# 📸 Histogram of Oriented Gradients (HOG) & Face Detection
A complete implementation of the Histogram of Oriented Gradients (HOG) descriptor and a template-based face detector using Normalized Cross-Correlation (NCC).

This project includes a full HOG pipeline, face-detection system, and visualization tools. Reference diagrams and algorithm explanations were adapted from computer vision learning materials:
- Project instructions
- HOG lecture slides
- Full implementation source code (p1.py)

---

## 🚀 Features

### ✔️ HOG Descriptor
Implements the full Dalal & Triggs style HOG pipeline:
- Sobel-based differential filtering
- Gradient magnitude + unsigned orientation
- Orientation binning into 6 bins
- 8×8 cells, 2×2 blocks, L2-normalized
- Final descriptor flattening

### ✔️ Face Detection
Sliding-window detector using HOG features + NCC:
- Extract HOG from a template image
- Compare with each target window using normalized cross-correlation
- Threshold detections
- Perform Non-Maximum Suppression (NMS) using IoU > 0.5

### ✔️ Visualizations
- HOG visualization using line renderings for orientation bins
- Face detection bounding boxes + confidence scores

---

## 🖼 Example Outputs

### HOG Visualization
(Insert your HOG figure here)

![HOG Visualization](images/hog_output.png)

### Face Detection Results
(Insert your detection figure here)

![Face Detection](images/face_detection.png)

---

## 🧠 Algorithm Overview

### 1. Differential Filtering
Using 3×3 Sobel filters:
- Horizontal derivative
- Vertical derivative

### 2. Gradient Magnitude & Orientation
Compute:
- ||∇I|| = sqrt(Ix² + Iy²)
- θ = atan2(Iy, Ix) mod π

### 3. Orientation Binning
Divide image into cells of size 8×8.
Accumulate gradient magnitudes into 6 bins covering:

[0°, 15°), [15°, 45°), …, [135°, 165°)

### 4. Block Normalization
For each 2×2 group of cells (stride 1), apply L2 normalization:

ĥᵢ = hᵢ / sqrt(Σⱼ hⱼ² + ε²)

with ε = 0.001.

### 5. Face Detection via NCC
For each sliding window:
- Extract HOG
- Subtract mean from both patch and template
- Compute normalized cross-correlation:

s = (a · b) / (||a|| ||b||)

Then apply thresholding and Non-Maximum Suppression to remove overlapping boxes.

---

## 📁 Repository Structure

```
.
├── p1.py               # Full implementation
├── images/             # Add your own output images here
│   ├── hog_output.png
│   ├── face_detection.png
├── assets/             # Template/target images (not included)
├── README.md
```

---

## ▶️ Running the Code

### Install dependencies
```
pip install numpy pillow matplotlib
```

### Run HOG & Face Detection
```
python p1.py
```

Outputs:
- HOG visualization of cameraman.tif
- Face detection bounding boxes on the target image

---

## 📚 References
- Project specification (algorithms, equations, and examples)
- HOG lecture slides
- Full implementation code (p1.py)
- Dalal & Triggs — Histograms of Oriented Gradients for Human Detection (CVPR 2005)

---

## ✨ Author
Adil Arya
