# DigitVision – Handwritten Digit Recognition Using CNN

## Project Overview
DigitVision is a deep learning project that recognizes handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN).

Users can draw a digit on the application screen, and the trained model predicts the digit.

## Technologies Used
- Python
- PyTorch
- CNN (Convolutional Neural Network)
- MNIST Dataset
- Tkinter
- Pillow
- Matplotlib

## Project Phases

### Phase 1: Data Collection and Preprocessing
- Downloaded the MNIST dataset.
- Loaded 60,000 training images and 10,000 testing images.
- Preprocessed and visualized handwritten digits.

### Phase 2: CNN Model Training
- Developed a CNN model using PyTorch.
- Trained the model on the MNIST dataset.
- Evaluated the model using testing data.
- Saved the trained model as `digit_cnn.pth`.

### Phase 3: GUI Application
- Developed a graphical application using Tkinter.
- Users can draw digits from 0 to 9.
- The CNN model predicts the handwritten digit.
- Displays prediction confidence and alternative predictions.

## How to Run

1. Clone the repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Train the model:

   `python train.py`

4. Run the application:

   `python app.py`

## Objective
To develop an AI-based handwritten digit recognition system using deep learning and CNN.