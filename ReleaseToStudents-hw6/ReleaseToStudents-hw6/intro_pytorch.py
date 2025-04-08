import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training=True):
    """
    Get the DataLoader for training or test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.FashionMNIST('./data', train=training, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=training)
    return loader

def build_model():
    """
    Build a simple neural network model.
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def build_deeper_model():
    """
    Build a deeper neural network model.
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    return model

def train_model(model, train_loader, criterion, T):
    """
    Train the model with the given training data.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    
    for epoch in range(T):
        correct = 0
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{len(train_loader.dataset)}({accuracy:.2f}%) Loss: {avg_loss:.3f}")

def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    Evaluate the model performance on the test set.
    """
    model.eval()
    correct = 0
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader.dataset)
    
    if show_loss:
        print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

def predict_label(model, test_images, index):
    """
    Predict and print the top 3 most likely class labels for an image.
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    model.eval()
    with torch.no_grad():
        logits = model(test_images)
        probabilities = F.softmax(logits, dim=1)
        top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
    
    for i in range(3):
        label = class_names[top3_indices[index][i].item()]
        prob = top3_probs[index][i].item() * 100
        print(f"{label}: {prob:.2f}%")

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    
    model = build_model()
    train_model(model, train_loader, criterion, T=5)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    
    test_images, _ = next(iter(test_loader))
    predict_label(model, test_images, 1)
