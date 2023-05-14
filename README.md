# COVID-19 detection and model optimization using Vision Transformers

> This is the codebase for Deep Learning Final-Project Spring 2023
> 
> **Team Members:** Mayuri Upadhyaya, Siddharth Nair, Charmee Mehta
>
> **Net IDs:** mbu2005, sm10437, cm6389

## Problem Statement
The coronavirus known as SARS-CoV-2 is the source of the illness COVID-19. Patients are screened at a medical facility using a PCR test as the first step in COVID-19 treatment. However, a medical professional makes a thorough diagnosis of the patient by examining a lung X-ray sample, which allows for the assessment of the lung damage and the progression of the infection. Because medical imaging is easy to use and quick, clinicians can diagnose illnesses and their effects more quickly. The literature has demonstrated that chest X-rays may be a source of testing for COVID-19 patients, however manually reviewing X-ray records is time-consuming and prone to error. We propose a Vision transformer model which is a deep learning pipeline for the detection of COVID-19 from chest X-ray based imaging. 

## Data
You can find the COVID-19 dataset here: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database.

It contains 6334 black-and-white images. The dataset contains four classes (a) Negative for Pneumonia, (b) Typical Appearance, (c) Intermediate Appearance, and (d) Atypical Appearance respectively.

## Model Description
Networks that work with data sequences are called transformers. These sequences are tokenized first, after which they are sent to the transformers. Transformers increase awareness (calculates pairwise inner product between each pair of the tokenized words.) An attention mechanism will expend (500^2)^2 operations on an image with a dimension of 500*500 pixels. As a result, rather than using global attention when studying images, academics typically use some type of local attention (cluster of pixels).

## Targets
We hope to adapt and scale the vision transformer to perform image classification on the COVID-19 radiography dataset. We will also try to improve the accuracy by performing hyperparameter tuning and comparing the variance of accuracies with different parameters.


## Steps to run:

Step 1: Install the following packages -
- Tensorflow
- Pytorch
- Matplotlib
- Numpy
- Pandas

Step 2: Clone this repository, open and run DL-Project.ipynb notebook.
