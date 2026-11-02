
import tkinter as tk
from tkinter import messagebox

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x



model = DigitCNN()

try:
    model.load_state_dict(
        torch.load(
            "digit_cnn.pth",
            map_location="cpu",
            weights_only=True
        )
    )
except FileNotFoundError:
    raise SystemExit(
        "digit_cnn.pth not found. Please run train.py first."
    )

model.eval()



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])



root = tk.Tk()
root.title("DigitVision - Handwritten Digit Recognition")
root.geometry("430x600")
root.resizable(False, False)

title = tk.Label(
    root,
    text="DigitVision",
    font=("Arial", 24, "bold")
)
title.pack(pady=(12, 4))

instruction = tk.Label(
    root,
    text="Draw any digit from 0 to 9",
    font=("Arial", 12)
)
instruction.pack(pady=4)



canvas_size = 280

canvas = tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black",
    cursor="cross",
    highlightthickness=2,
    highlightbackground="gray"
)
canvas.pack(pady=12)

drawing = Image.new("L", (canvas_size, canvas_size), 0)
draw = ImageDraw.Draw(drawing)



def paint(event):
    x, y = event.x, event.y
    radius = 9

    canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="white",
        outline="white"
    )

    draw.ellipse(  
        (x - radius, y - radius, x + radius, y + radius),
        fill=255
    )


canvas.bind("<B1-Motion>", paint)
canvas.bind("<Button-1>", paint)



def predict_digit():
    bbox = drawing.getbbox()

    if bbox is None:
        messagebox.showinfo(
            "DigitVision",
            "Please draw a digit first!"
        )
        return

    
    cropped = drawing.crop(bbox)

    
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)


    img = Image.new("L", (28, 28), 0)

    x = (28 - cropped.width) // 2
    y = (28 - cropped.height) // 2

    img.paste(cropped, (x, y))

   
    image_tensor = transform(img).unsqueeze(0)

    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_digit = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_digit].item() * 100

        
        top_values, top_indices = torch.topk(probabilities, 3)

    result_label.config(
        text=(
            f"Predicted Digit: {predicted_digit}\n"
            f"Model Confidence: {confidence:.1f}%"
        )
    )

    alternatives = "Other predictions: "

    for i in range(1, 3):
        digit = top_indices[0, i].item()
        score = top_values[0, i].item() * 100
        alternatives += f"{digit} ({score:.1f}%)  "

    alternatives_label.config(text=alternatives)



def clear_canvas():
    global drawing, draw

    canvas.delete("all")

    drawing = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(drawing)

    result_label.config(
        text="Your prediction will appear here"
    )

    alternatives_label.config(text="")


predict_button = tk.Button(
    root,
    text="Predict Digit",
    command=predict_digit,
    font=("Arial", 12, "bold"),
    width=16,
    bg="#1976D2",
    fg="white",
    activebackground="#1565C0"
)
predict_button.pack(pady=5)

clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 11),
    width=16
)
clear_button.pack(pady=4)



result_label = tk.Label(
    root,
    text="Your prediction will appear here",
    font=("Arial", 14, "bold"),
    justify="center"
)
result_label.pack(pady=(10, 4))

alternatives_label = tk.Label(
    root,
    text="",
    font=("Arial", 10),
    wraplength=400
)
alternatives_label.pack(pady=4)



root.mainloop()
