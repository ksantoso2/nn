# module import
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import FashionMNIST
import timm 

# Load data
CaiT_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
torch.manual_seed(47)

CaiT_dataset = FashionMNIST(root="./data", train=True, download=True, transform=CaiT_transforms)
CaiT_test_dataset = FashionMNIST(root="./data", train=False, download=True, transform=CaiT_transforms)

train_size = int(0.9 * len(CaiT_dataset))
val_size = len(CaiT_dataset) - train_size
CaiT_train_dataset, CaiT_valid_dataset = random_split(CaiT_dataset, [train_size, val_size])
batch_size = 256

CaiTTrainLoader = DataLoader(CaiT_train_dataset, batch_size=256,
                                          shuffle=True)
CaiTValidLoader = DataLoader(CaiT_valid_dataset, batch_size=256,
                                          shuffle=False)
CaiTTestLoader = DataLoader(CaiT_test_dataset, batch_size=256,
                                          shuffle=False)

# Load CaiT model (no pretrained weights for grayscale FashionMNIST)
model = timm.create_model("cait_xxs24_224", pretrained=False, num_classes=10)
model = model.to(torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Eval
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if device.type == "mps":
            loss = loss.to("cpu")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct

EPOCHS = 10
train_losses = []
valid_losses = []
train_correct = []
valid_correct = []

for epoch in range(EPOCHS):
    t_loss, t_correct = train(model, CaiTTrainLoader, criterion, optimizer)
    v_loss, v_correct = evaluate(model, CaiTValidLoader, criterion)

    train_losses.append(t_loss)
    valid_losses.append(v_loss)
    train_correct.append(t_correct)
    valid_correct.append(v_correct)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Valid Loss: {v_loss:.4f} | "
          f"Train Acc: {t_correct/len(CaiTTrainLoader.dataset)*100:.2f}% | Valid Acc: {v_correct/len(CaiTValidLoader.dataset)*100:.2f}%")

test_loss, test_acc = evaluate(model, CaiTTestLoader, criterion)
print(f"✅ Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")

import matplotlib.pyplot as plt

def plot_results(train_losses, valid_losses, train_correct, valid_correct, epochs, trainLoader, validLoader):
    # Convert losses from tensors to floats (if needed)
    train_losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    valid_losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in valid_losses]

    # Convert correct counts to accuracy percentages
    train_acc = [correct / len(trainLoader.dataset) * 100 for correct in train_correct]
    valid_acc = [correct / len(validLoader.dataset) * 100 for correct in valid_correct]

    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label="Train Accuracy")
    plt.plot(epochs_range, valid_acc, label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_results(train_losses, valid_losses, train_correct, valid_correct, EPOCHS, CaiTTrainLoader, CaiTValidLoader)
