import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import pickle


# Download CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
x_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True).data

minority_class = 3  # 'cat' class

minority_data = [(image, label) for image, label in trainset if label == minority_class]
majority_data = [(image, label) for image, label in trainset if label != minority_class]

# Let's imbalance the 'cat' class by taking only 500 instances
minority_data = minority_data[:500]

imbalanced_data = minority_data + majority_data

# Save imbalanced dataset
with open('imbalanced_data.pkl', 'wb') as f:
    pickle.dump(imbalanced_data, f)

with open('test_data.pkl', 'wb') as f:
    pickle.dump(x_test, f)

print(f"Imbalanced dataset created with {len(imbalanced_data)} samples. Data saved to 'imbalanced_data.pkl'")
print(f"Test dataset created with {len(x_test)} samples. Data saved to 'test_data.pkl'")
