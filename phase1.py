
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Step 1: Convert images into tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Step 2: Download the MNIST training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Step 3: Download the MNIST testing dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Step 4: Check dataset size
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# Step 5: Display 10 handwritten digit images
plt.figure(figsize=(12, 4))

for i in range(10):
    image, label = train_dataset[i]

    plt.subplot(2, 5, i + 1)
    plt.imshow(image.squeeze(0), cmap="gray")
    plt.title(f"Digit: {label}")
    plt.axis("off")

plt.tight_layout()
plt.show()

