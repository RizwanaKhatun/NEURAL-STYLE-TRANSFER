# NEURAL-STYLE-TRANSFER

**COMPANY**: CODTECHIT SOLUTIONS

**NAME**: SHAIK RIZWANA KHATUN

**INTERN ID**: CT08DHQ

**DOMAIN**: ARTIFICIAL INTELLIGENCE

**BATCH DURATION**: DEC12TH,2024 to JAN 12TH,2025

**OVERVIEW OF THE TASK**

Overview of the Neural Style Transfer Program
The above program implements Neural Style Transfer (NST) using a pre-trained VGG19 model. Neural Style Transfer blends two images: a content image (e.g., a photograph) and a style image (e.g., a painting or artwork), to create a new image that retains the content of the original photograph while applying the artistic style from the artwork. Here's how the program works:

Steps in the Program
Image Preprocessing:

The content and style images are loaded and preprocessed. Images are resized to a manageable size (max 400 pixels wide) while maintaining their aspect ratios. The images are then normalized based on ImageNet's mean and standard deviation values to ensure consistency with the pre-trained VGG19 model.
VGG19 Model:

The program uses the VGG19 model pre-trained on the ImageNet dataset. The VGG19 model extracts both content and style features from the images.
The model is used without its classification layers (since we only need feature extraction).
Feature Extraction:

Content features: These are taken from the content image and represent the structure and details of the photograph.
Style features: These are extracted from the style image and represent the texture, color, and other artistic elements of the style.
Loss Functions:

Content Loss: Measures how much the target image’s content differs from the content image.
Style Loss: Measures how much the target image’s style differs from the style image. This is done using Gram matrices to capture the correlations between different feature channels.
Optimization:

The target image (which starts as a copy of the content image) is iteratively adjusted by gradient descent to minimize the total loss, which is a weighted sum of the content and style losses.
The optimizer used is LBFGS, a popular optimizer for image generation tasks.


**OUTPUT**:
