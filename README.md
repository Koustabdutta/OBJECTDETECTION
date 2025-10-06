# 🚁 Real-Time UAV Object Detection and Classification (VisDrone + YOLOv8)

This project demonstrates **real-time object detection and classification** on UAV (drone) imagery using the [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset) and **YOLOv8**.
It is designed for efficient execution on a **desktop GPU** (RTX 5070, 12 GB) and can later be deployed on **FPGA** hardware (e.g. Xilinx Vitis/Vivado).

The notebook and scripts allow you to:

* Detect and classify objects (cars, pedestrians, cyclists, etc.) in **real-time video**.
* Benchmark multiple pretrained lightweight models (YOLOv8n, MobileNet, ViT-Tiny).
* Visualize detections with bounding boxes, confidence scores, and FPS overlay.
* Profile inference speed, GPU memory, and latency for future FPGA integration.

---

## 📸 Features

✅ **YOLOv8n** real-time detection (2.6 M params, anchor-free).
✅ **MobileNetV3** and **ViT-Tiny** classification backends for cropped objects.
✅ Live **video/webcam inference** with OpenCV.
✅ **Visualization overlay** (bounding boxes, labels, confidence, FPS).
✅ GPU/CPU fallback for flexible deployment.
✅ Designed for **FPGA quantization and deployment** using Xilinx Vitis AI.

---

## 🧠 Model Summary

| Model             | Task             | Params | FPS (RTX 5070 GPU) | Notes                            |
| :---------------- | :--------------- | :----- | :----------------- | :------------------------------- |
| YOLOv8n           | Object Detection | 2.6 M  | ~60–100            | COCO-pretrained, VisDrone-ready  |
| MobileNetV3-Small | Classification   | 2.55 M | ~120               | Fast, compact                    |
| ViT-Tiny          | Classification   | 5 M    | ~70                | Transformer-based, high accuracy |

All models are pretrained and **require no additional training** for demonstration.

---

## 🗂️ Dataset

* **VisDrone2019-DET** — Real-world drone imagery with 10,209 images and 2.6 M labeled bounding boxes.
* Convert original annotations to YOLO format using Ultralytics’ helper script:

  ```bash
  yolo data convert visdrone
  ```
* Folder structure:

  ```
  VisDrone/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/
  ```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/real-time-drone-detection.git
cd real-time-drone-detection

# Install dependencies
pip install ultralytics timm opencv-python torch torchvision tqdm scikit-learn
```

---

## 🚀 Usage

### 1️⃣ Run the Real-Time Detection Notebook

Launch the Jupyter notebook:

```bash
jupyter notebook RealTime_Drone_Detection.ipynb
```

or run as a standalone script:

```bash
python realtime_yolo.py
```

### 2️⃣ Configure the Video Source

Open the notebook/script and set:

```python
VIDEO_SOURCE = 0  # 0 for webcam or "path/to/video.mp4"
```

Press `q` to exit.

---

## 🧩 Example Output

Bounding boxes and labels are drawn on live video frames:

```
FPS: 75.2  AvgFPS: 72.5  Detections: 12
```

Each detected object is shown with its class name and confidence:
![example\_output](docs/example_output.jpg)

---

## 📊 Evaluation Metrics

* **Detection:** mAP@50, mAP@50:95 (via `model.val()`).
* **Classification:** Accuracy, F1-score (via scikit-learn).
* **Efficiency:** FPS, latency, and GPU memory profiling.
* **FPGA Readiness:** Simulated latency and memory footprint for quantized models.

---

## 🧮 FPGA Deployment (Future Work)

Once validated on GPU:

1. Export trained models to ONNX:

   ```bash
   yolo export format=onnx
   ```
2. Quantize to 8-bit fixed-point with **Vitis AI**:

   ```bash
   vai_q_pytorch quantize.py --model yolov8n.onnx --quant_mode calib
   ```
3. Compile for your FPGA target (e.g., ZCU104, Alveo):

   ```bash
   vai_c_xir -x yolov8n_quantized.xmodel -a arch.json -o build
   ```
4. Deploy via DPU runner or XRT.

---

## 🧰 Hardware & Environment

* **CPU:** AMD Ryzen 9 7900
* **GPU:** Inno3D RTX 5070 12 GB
* **RAM:** 64 GB
* **OS:** Ubuntu 22.04 / Windows 11
* **Python:** 3.10+
* **Frameworks:** PyTorch, Ultralytics YOLOv8, timm, OpenCV

---

## 🧪 Example Performance (RTX 5070)

| Video             | Resolution | Avg FPS | GPU Mem | Comments         |
| ----------------- | ---------- | ------- | ------- | ---------------- |
| Webcam            | 720p       | 98 FPS  | 2.1 GB  | Stable real-time |
| VisDrone test.mp4 | 1080p      | 65 FPS  | 3.4 GB  | Smooth detection |
| Drone synthetic   | 480p       | 130 FPS | 1.6 GB  | Extremely fast   |

---

## 📚 References

* [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
* [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
* [Vitis AI Quantization Guide](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai)
* [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
* [ViT Paper (Dosovitskiy et al.)](https://arxiv.org/abs/2010.11929)

---

## 🧑‍💻 Author

**Koustab Dutta**
M.Sc. Student, Dept. of Data Science & AI, Ramakrishna Mission Residential College, Narendrapur
📧 *[koustabdutta@example.com](mailto:koustabdutta@example.com)*

---

## 🪄 License

MIT License © 2025 Koustab Dutta.
You are free to use, modify, and distribute this project with attribution.

---

**⭐ If you find this project helpful, consider giving it a star on GitHub!**
