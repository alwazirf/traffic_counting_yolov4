# yolov4-custom-functions
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Getting Started
### Conda (Recommended)
```bash
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Downloading My Pre-trained Weights
USE MY FINAL CUSTOM TRAINED CUSTOM WEIGHTS: https://drive.google.com/file/d/1EYKcptLtDeLJJNhYKUpyOKqWdzhkopk4/view?usp=sharing

Copy and paste customfinals .weights file into the 'data' folder

## YOLOv4 Using Tensorflow (tf, .pb model)
To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.
```bash
# Convert darknet weights to tensorflow
## custom
python save_model.py --weights ./data/customfinals.weights --output ./checkpoints/customfinals-416 --input_size 416 --model yolov4 

# Run yolov4 tensorflow model
python main.py
```

### References  

   Huge shoutout goes to hunglc007 for creating the backbone of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
