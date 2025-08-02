# Step 1: Load CIFAR-10 and Preprocess
import torch
import torchvision
import torchvision.transforms as transforms
import time
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)

# Step 2: Define a Simple CNN Model
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Student model for distillation
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Knowledge distillation loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        loss_kl = self.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        loss_ce = self.ce(student_logits, labels)
        return self.alpha * loss_ce + (1 - self.alpha) * loss_kl

# Train student model with distillation
def train_distilled_model(teacher_model, student_model, trainloader, testloader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    criterion = DistillationLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    teacher_model.eval()

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            loss = criterion(student_outputs, teacher_outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Distillation] Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")

    print("Distillation training finished.")
    return student_model

# Train teacher model and save it
teacher_model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)

print("Training teacher model...")
teacher_model.train()
for epoch in range(10):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = teacher_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Teacher] Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")

torch.save(teacher_model.state_dict(), "baseline_model.pth")
print("Teacher model saved.")


# Load teacher model and train student
teacher_model = SimpleCNN()
teacher_model.load_state_dict(torch.load("baseline_model.pth"))
student_model = SmallCNN()
student_model = train_distilled_model(teacher_model, student_model, trainloader, testloader)
torch.save(student_model.state_dict(), "distilled_model.pth")
print("Distilled model saved.")

# Benchmark distilled model
def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def measure_inference_time(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)
    end_time = time.time()
    return end_time - start_time

print("Evaluating distilled model...")
student_model.load_state_dict(torch.load("distilled_model.pth"))
distilled_acc = evaluate(student_model, testloader)
distilled_time = measure_inference_time(student_model, testloader)
distilled_size = os.path.getsize("distilled_model.pth") / 1e6

print(f"Distilled Accuracy: {distilled_acc:.2f}%")
print(f"Distilled Inference Time: {distilled_time:.2f} sec")
print(f"Distilled Model Size: {distilled_size:.2f} MB")
