import torch
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

# Load your imbalanced data using pickle
with open('imbalanced_data.pkl', 'rb') as f:
    imbalanced_data = pickle.load(f)

labels = [label for _, label in imbalanced_data]
class_counts = Counter(labels)

# Identify the class you want to augment
target_class = 3

# Collect indices of samples from the target class
target_indices = [i for i, label in enumerate(labels) if label == target_class]

# Load existing images and labels
images, labels = zip(*imbalanced_data)

# Create a subset of samples from the target class
target_samples = [images[i] for i in target_indices]

# Load all samples (not just the augmented class)
all_samples = [images[i] for i in range(len(images))]

# Define data augmentation transformations
desired_image_size = 32  # Desired image size
transform = transforms.Compose([
                                transforms.Resize((desired_image_size, desired_image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                transforms.RandomRotation(degrees=1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                ])

# Convert the images to tensors and apply data augmentation
augmented_samples = [transform(Image.fromarray(np.uint8(image))) for image in target_samples]
all_samples = [transform(Image.fromarray(np.uint8(image))) for image in all_samples]

# Convert the labels to a tensor
augmented_labels = torch.tensor([target_class] * len(augmented_samples))
all_labels = torch.tensor(labels)  # Use all labels, not just the augmented class

# Create the TensorDataset using the augmented data and labels
train_dataset = TensorDataset(torch.stack(all_samples), all_labels)

# Create a DataLoader for the training data
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Collect all labels from the training loader
all_labels = []
for batch in train_loader:
    _, labels = batch
    all_labels.extend(labels.tolist())

# Convert the labels to a tensor PyTorch
train_labels_tensor = torch.tensor(all_labels)

# Count the number of examples for each class
class_labels, class_counts = torch.unique(train_labels_tensor, return_counts=True)

# Convert to Numpy arrays for visualization
class_labels = class_labels.numpy()
class_counts = class_counts.numpy()

# Plot the distribution of classes
plt.bar(class_labels, class_counts)
plt.xlabel('Classe')
plt.ylabel('Nombre d\'exemples')
plt.title('Distribution des Classes dans les Données d\'Entraînement')
plt.xticks(class_labels)
plt.show()

# Affichage des détails
for label, count in zip(class_labels, class_counts):
    print(f"Classe {label}: {count} exemples")
# Define the CNNModel architecture
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (desired_image_size // 4) * (desired_image_size // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (desired_image_size // 4) * (desired_image_size // 4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the CNNModel and move it to the GPU if available
num_classes = len(class_counts)
model = CNNModel(num_classes=num_classes)
print(model)  # Imprime les détails du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)

# Train the model
num_epochs = 50
losses = []
# Train the model 
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:  
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

print("Training completed!")
# Enregistrement du modèle
torch.save(model.state_dict(), 'trained_model.pth')

# Enregistrement des pertes d'entraînement
with open('losses.pkl', 'wb') as f:
    pickle.dump(losses, f)
output_folder = "saved_images"  # Dossier où les images seront sauvegardées
os.makedirs(output_folder, exist_ok=True)  # Création du dossier s'il n'existe pas
num_images = len(augmented_samples)
max_subplot = min(num_images, 10)  # Limiter le nombre de sous-tracés à 10 au maximum
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(12, 12))
# Afficher les images originales
for i in range(max_subplot):
    original_image = Image.fromarray(np.uint8(target_samples[i]))
    ax = plt.subplot(max_subplot, 2, 2 * i + 1)
    ax.imshow(original_image)
    ax.set_title(f'Image d\'origine - Classe : {class_names[labels[i]]}')
    ax.axis('off')
    plt.delaxes(ax)  # Supprimer explicitement les axes superposés

# Afficher les images augmentées
for i in range(max_subplot):
    augmented_image = augmented_samples[i].numpy().transpose((1, 2, 0))
    augmented_image = np.clip(augmented_image, 0, 1)
    ax = plt.subplot(max_subplot, 2, 2 * i + 2)
    ax.imshow(augmented_image)
    ax.set_title(f'Image augmentée - Classe : {class_names[labels[i]]}')
    ax.axis('off')
    plt.delaxes(ax)  # Supprimer explicitement les axes superposés

plt.tight_layout()
plt.show()
# Plot the loss curve
plt.figure()
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
# Load test data
with open('test_data.pkl', 'rb') as f:
    test_samples = pickle.load(f)

# Load true test labels
with open('true_test_labels.pkl', 'rb') as f:
    true_test_labels = pickle.load(f)

# Define data augmentation transformations
desired_image_size = 32
transform = transforms.Compose([
                                transforms.Resize((desired_image_size, desired_image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                ])

# Convert test images to tensors and apply transformations
test_samples = [transform(Image.fromarray(np.uint8(image))) for image in test_samples]

# Create the TensorDataset for test data
test_dataset = TensorDataset(torch.stack(test_samples))

# Create a DataLoader for the test data
test_batch_size = 10
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
# Evaluate the model on test data
model.eval()
predicted_labels = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0].to(device)  # Move inputs to device
        
        # Calculate the scores (output of the model)
        scores = model(inputs)
        
        # Apply Softmax to get probabilities
        probabilities = F.softmax(scores, dim=1)
        
        # Select the predicted class (the one with the highest probability)
        _, predicted = torch.max(probabilities, 1)
        
        # Add predictions to the list
        predicted_labels.extend(predicted.cpu().numpy())
# Calculate accuracy
total_examples = len(true_test_labels)
correct_predictions = sum(1 for pred, true in zip(predicted_labels, true_test_labels) if (pred == true).any())
accuracy = (correct_predictions / total_examples) * 100
# Calculate class distribution in test data
class_labels, class_counts = np.unique(true_test_labels, return_counts=True)
# Enregistrement des prédictions dans un fichier pkl
with open('predictions.pkl', 'wb') as f:
    pickle.dump(predicted_labels, f)
# Plot the distribution of classes in test data
plt.bar(class_labels, class_counts)
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.title('Distribution of Classes in Test Data')
plt.xticks(class_labels)
plt.show()
print(f"Total examples: {total_examples}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")

