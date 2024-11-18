import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import numpy as np
from datetime import datetime
import os

class ImageClassifier:
    def __init__(self, batch_size=128, num_epochs=10, learning_rate=0.001):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize transforms for data preprocessing
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # CIFAR-10 classes
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        """Set up CIFAR-10 data loaders"""
        # Download and load CIFAR-10 dataset
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform_train)
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform_test)
        self.testloader = DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
    def setup_model(self):
        """Set up pre-trained ResNet model"""
        # Load pre-trained ResNet18 model
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Modify the first conv layer to handle CIFAR-10's 32x32 images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool as we have smaller images
        
        # Modify final fully connected layer for 10 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        
        # Move model to GPU if available
        self.model = self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(self.trainloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Batch [{i+1}/{len(self.trainloader)}] '
                      f'Loss: {running_loss/100:.3f} '
                      f'Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
                
        return running_loss / len(self.trainloader), 100. * correct / total
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return val_loss / len(self.testloader), 100. * correct / total
    
    def train(self):
        """Full training loop"""
        best_acc = 0
        print(f"Training on {self.device}")
        
        for epoch in range(self.num_epochs):
            print(f'\nEpoch: {epoch+1}/{self.num_epochs}')
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - start_time
            
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
            print(f'Epoch Time: {epoch_time:.2f}s')
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_acc:
                print('Saving model...')
                state = {
                    'model': self.model.state_dict(),
                    'acc': val_acc,
                    'epoch': epoch,
                }
                if not os.path.exists('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/best_model.pth')
                best_acc = val_acc
    
    def load_best_model(self):
        """Load the best model from checkpoint"""
        checkpoint = torch.load('./checkpoint/best_model.pth')
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loaded model with accuracy: {checkpoint['acc']:.2f}%")
    
    def predict(self, image_tensor):
        """Make prediction for a single image"""
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            _, predicted = output.max(1)
            return self.classes[predicted.item()]

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Starting training run at {timestamp}")
    
    # Initialize and train the model
    classifier = ImageClassifier(
        batch_size=128,
        num_epochs=10,
        learning_rate=0.001
    )
    
    # Print model summary
    print("\nModel Architecture:")
    print(classifier.model)
    
    # Print GPU information if available
    if torch.cuda.is_available():
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    # Train the model
    classifier.train()
    
    # Load best model and make a test prediction
    classifier.load_best_model()
    
    # Test prediction on a single image
    dataiter = iter(classifier.testloader)
    images, labels = next(dataiter)
    prediction = classifier.predict(images[0])
    print(f"\nExample prediction: {prediction}")

if __name__ == "__main__":
    main()