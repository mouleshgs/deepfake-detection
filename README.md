# Deepfake Image Detection using VGG16

## Project Overview

This project is a deepfake image detection system that utilizes the VGG16 model to classify images as real or fake. The application is built using Flask, allowing users to upload images and receive predictions on their authenticity.

## Directory Structure
```
├── .gitignore
├── README.md
├── app.py                 
├── requirement.txt       
├── static                 
│   ├── favicon.ico        
│   ├── logo.png           
│   ├── styles
│   │   └── main.css       
│   └── uploads            
│       ├── deep r.jpg
│       ├── fake_8467.jpg
│       ├── fake_8475.jpg
│       └── real r.jpg
├── templates             
│   ├── about.html
│   ├── base.html
│   ├── contact.html
│   ├── index.html
│   └── result.html
├── test_model.py          
└── train_model.py         
```
# Installation

## Prerequisites

Python 3.x
Virtual environment (optional but recommended)

## Steps

### Clone the repository:
```bash
git clone https://github.com/mouleshgs/deepfake-detection
cd deepfake-detection
```

### Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies:
```
pip install -r requirement.txt
```
## Usage

### Training the Model

Run the following command to train the deepfake detection model using VGG16:

`python train_model.py`

### Testing the Model

To test the trained model:

`python test_model.py`

Running the Flask Application

### Start the Flask server:

`python app.py`

Then, open a browser and visit `http://127.0.0.1:5000/` to access the web interface.

## Model Details

* Architecture: **VGG16**
* Pretrained Weights: **ImageNet**
* Customization: The model is fine-tuned with additional dense layers for binary classification (real vs. fake images).

## Features

Upload an image via the web interface.
Receive a prediction on whether the image is real or fake.
View model results and details.
