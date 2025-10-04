# AIRL-Assignment

# Q1 — Vision Transformer on CIFAR-10 (PyTorch)

## Goal
Implement a Vision Transformer (ViT) and train it on the CIFAR-10 dataset (10 classes) using PyTorch. The objective is to achieve the highest possible test accuracy. The project is designed to run **entirely on Google Colab**.

Paper Reference: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., ICLR 2021.

## How to Run in Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload or mount this project repository.
3. Ensure you have the required packages installed:
   **!pip install torch torchvision matplotlib**
Run the main notebook/script:
  **!python vit_cifar10.py**


The notebook will automatically download CIFAR-10, train the ViT model, and display training/test accuracy.

Model Configuration (Best Model)
**Parameter	            Value**
Image size	             32x32
Patch size	              4x4
Embedding dimension	      128
Number of heads (MHSA)	   8
Number of encoder layers	 6
MLP hidden dimension	    256
Dropout                  	0.1
Optimizer	               AdamW
Learning rate schedule	Cosine Annealing
Batch size	              128
Number of epochs	         50
Results
Metric	Value
Test Accuracy (%)	89.5


# Q2 — Text-Driven Image Segmentation with SAM 2

## Overview
Segment a single object in an image based on a **text prompt** using **SAM 2**.
The pipeline uses **CLIPSeg** to generate seed points and SAM 2 to produce the final mask overlay.

## Pipeline
Upload Image → Text Prompt → CLIPSeg Seeds → SAM 2 → Mask Overlay → Final Output

* **Upload Image:** Input image via Colab.
* **Text Prompt:** Describe the object (e.g., “a dog”).
* **CLIPSeg Seeds:** Convert soft mask into seed points.
* **SAM 2:** Predict object mask.
* **Overlay:** Semi-transparent red mask on original image.

## How to Run
1. Open `q2.ipynb` in **Google Colab**.
2. Run all cells to install dependencies.
3. Upload an image and enter a text prompt.
4. View **Original Image** and **Segmentation Overlay**.

## Dependencies
* Python 3.x, PyTorch
* Transformers (CLIPSeg)
* Segment Anything (`segment-anything`)
* OpenCV, Pillow, Matplotlib

## Limitations
* Best for **single-object segmentation**.
* Mask quality depends on **text prompt clarity**.
* Complex backgrounds may reduce accuracy.
