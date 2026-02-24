# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Include the problem statement and Dataset


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.

### STEP 3: 
Visualize sample images from the dataset.

### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.



## PROGRAM

### Name:MANIKANDAN T

### Register Number:212224110037

```python
# Load Pretrained Model and Modify for Transfer Learning
model=models.vgg19(weights=VGG19_Weights.DEFAULT)



# Modify the final fully connected layer to match the dataset classes
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)



# Include the Loss function and optimizer
criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)



# Train the model
from torch.nn.modules import loss
## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss  = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))
        model.train()
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

 
    print("Name:MANIKANDAN T")
    print("Register Number:212224110037")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_model(model, train_loader,test_loader)


```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot


<img width="732" height="721" alt="image" src="https://github.com/user-attachments/assets/a5670905-3a8f-43d9-b2a6-987d3c185193" />


## Confusion Matrix

<img width="760" height="650" alt="image" src="https://github.com/user-attachments/assets/c0e31c2c-f135-4c53-8e85-85a81681a2b2" />


## Classification Report

<img width="568" height="226" alt="image" src="https://github.com/user-attachments/assets/b9c37ea3-c35c-45e8-8728-e6ab64f66e02" />


### New Sample Data Prediction

<img width="650" height="558" alt="image" src="https://github.com/user-attachments/assets/1dea63dc-cb0e-4c98-9ab8-b8ef32b361c0" />


## RESULT

Thus, the image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
