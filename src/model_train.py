import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os
from src.app_state import ModelTraining, Question
import time

class CatBreedDataset(Dataset):
    def __init__(self, image_path, class_names, transform):
        self.image_path = image_path
        self.class_names = class_names
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        self.transform = transform
        self.image_files = []
        self.image_labels = []
        self.image_tensor = []
        for class_name in self.class_names:
            for image_file in os.listdir(os.path.join(self.image_path, class_name)):
                self.image_files.append(os.path.join(self.image_path, class_name, image_file))
                self.image_labels.append(self.class_to_idx[class_name])
                self.image_tensor.append(self.transform(Image.open(os.path.join(self.image_path, class_name, image_file)).convert('RGB')))
                
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_tensor = self.image_tensor[idx]
        image_label = self.image_labels[idx]
        return image_tensor, image_label

class CatBreedClassifier(nn.Module):
    def __init__(self):
        super(CatBreedClassifier, self).__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 112 x 112
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 56 x 56
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 28 x 28
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 512 x 14 x 14
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Changed from 1024 to 512
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)   # Output: 512 x 7 x 7
        )
        
        # Calculate the flattened size: 512 channels * 7 * 7 = 25088
        self.fc_block = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 7 * 7, 512),  # Adjusted input size
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 15),  # 15 breeds
            torch.nn.LogSoftmax(dim=1)
        )
        
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x
    
def initial_model_training(existing_model_trainings: list[ModelTraining] | None) -> list[ModelTraining]:
    if existing_model_trainings is None:
        existing_model_trainings = []
        
    model_trainings = [
        ModelTraining("Classification Model", "classification_model.pth", get_classification_model),
        ModelTraining("ResNet Model", "resnet_model.pth", get_resnet_model),
        ModelTraining("EfficientNet Model", "efficientnet_model.pth", get_efficient_net_model),
        ModelTraining("VGG Model", "vgg_model.pth", get_vgg_model)
    ]
    
    for model_training in model_trainings:
        if model_training not in existing_model_trainings:
            existing_model_trainings.append(model_training)
        else:  
            existing_model_training = existing_model_trainings[existing_model_trainings.index(model_training)]
            existing_model_training.get_model_function = model_training.get_model_function
    existing_model_trainings.sort(key=lambda x: x.model_name)
    return existing_model_trainings
     

def get_cat_breed_dataset(app_state, transform) -> CatBreedDataset: 
    return CatBreedDataset(app_state.image_path, app_state.class_names, transform)


def get_transformation() -> transforms.Compose:
    return transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop((224, 224)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(10),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])]
    )
    
def get_classification_model() -> nn.Module:
    return CatBreedClassifier()

def get_efficient_net_model() -> nn.Module:
    eff_model =  torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    for param in eff_model.parameters():
        param.requires_grad = False
    eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, 15)
    return eff_model

def get_resnet_model() -> nn.Module:
    res_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    for param in res_model.parameters():
        param.requires_grad = False
    res_model.fc = nn.Linear(res_model.fc.in_features, 15)
    return res_model

def get_vgg_model() -> nn.Module:
    vgg_model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    for param in vgg_model.parameters():
        param.requires_grad = False
    vgg_model.classifier[6] = nn.Linear(vgg_model.classifier[6].in_features, 15)
    return vgg_model

def train_model(app_state,model: nn.Module, model_name: str) -> tuple[list[float], list[float], list[float], list[float], float]:
    print(f"Training model on {app_state.device}")
    transform = get_transformation()
     
    print(f"Loading dataset")
    dataset = get_cat_breed_dataset(app_state, transform)
    
    # Split dataset into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders for both sets
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    print(f"Dataset split: {train_size} training samples, {test_size} test samples")
    
    print(f"Loading model")
    model.to(app_state.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    
    start_time = time.time()
    for epoch in range(5):
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(app_state.device)
            labels = labels.to(app_state.device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}")
            
        training_loss.append(running_loss / len(train_loader))
        training_accuracy.append(correct / total)
            
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                inputs = inputs.to(app_state.device)
                labels = labels.to(app_state.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) 
                correct += (predicted == labels).sum().item()
            
            validation_loss.append(running_loss / len(test_loader))
            validation_accuracy.append(correct / total)
            
        print(f"Accuracy of the network on the {total} test images: {100 * correct / total}%")
    print(f"Finished Training")
    print(f"Loss: {running_loss / len(train_loader)}")
    training_time = time.time() - start_time
    print(f"Saving model to {model_name}")
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, model_name)
    
    return training_loss, training_accuracy, validation_loss, validation_accuracy, training_time


def infer_model(app_state, model_name: str, question: Question) -> str | None:
    
    model_training = next((model_training for model_training in app_state.model_trainings if model_training.model_name == model_name), None)
    if model_training is None:
        raise ValueError(f"Model {model_name} not found")
    
    model = torch.jit.load(model_training.model_file_name) 
    
    transform = get_transformation()
    image = transform(Image.open(question.image_path).convert('RGB'))
    image = image.unsqueeze(0) 
    image = image.to(app_state.device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return app_state.class_names[predicted.item()] 
    
