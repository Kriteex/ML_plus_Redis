import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import redis
import numpy as np
import pickle








# Initialize Redis
r = redis.Redis(host='localhost', port=6379, db=0)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Neural network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()



try:
    # Načtení z Redisu
    loaded_serialized_state_dict = r.get("model_state_dict")
    loaded_state_dict = pickle.loads(loaded_serialized_state_dict)
except:
    print("No saved weights found.")

# Načtení do modelu
try:
    net.load_state_dict(loaded_state_dict)
    print("Váhy úspěšně načteny.")
except Exception as e:
    print(f"Chyba při načítání vah: {e}")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop with integrated Redis storage
for epoch in range(1):
    for i, data in enumerate(trainloader, 0):
        net.train()
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 2000 == 1999:
            for name, param in net.named_parameters():

                # Uložení do Redisu
                state_dict = net.state_dict()
                serialized_state_dict = pickle.dumps(state_dict)
                r.set("model_state_dict", serialized_state_dict)
                print(f"Saving {name} with shape {param.shape}")
                
                #r.set(name, param.detach().cpu().numpy().tobytes())
                # Save tensor shape as metadata
                #r.set(f"{name}_shape", str(param.shape))

        # Print statistics
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}")
            # Set the model to evaluation mode
            net.eval()

            correct = 0
            total = 0

            # No gradient computation is needed during inference
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    outputs = net(inputs)
                    
                    # Get predicted labels from max value
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Total number of labels
                    total += labels.size(0)
                    
                    # Total correct predictions
                    correct += (predicted == labels).sum().item()

            # Calculate accuracy
            accuracy = 100 * correct / total

            print(f'Accuracy of the model on test images: {accuracy}%')

