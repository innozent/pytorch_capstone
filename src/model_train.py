import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os
from src.app_state import ModelTraining, Question, AppState
import time 
import numpy as np

class Models:
    ClassificationModel = "Classification Model"
    ResNetModel = "ResNet Model"
    EfficientNetModel = "EfficientNet Model"
    VGGModel = "VGG Model" 

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
        ModelTraining(Models.ClassificationModel, "classification_model.pth", get_classification_model),
        ModelTraining(Models.ResNetModel, "resnet_model.pth", get_resnet_model),
        ModelTraining(Models.EfficientNetModel, "efficientnet_model.pth", get_efficient_net_model),
        ModelTraining(Models.VGGModel, "vgg_model.pth", get_vgg_model)
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


def infer_model(app_state: AppState, model_name: str, question: Question) -> str | None:
    
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
    
def grad_cam(app_state, model_name: str, question: Question) -> tuple[Image.Image, str]:
    """
    Implements GradCAM visualization for model interpretability without using OpenCV.
    Works with any model architecture by dynamically finding the appropriate convolutional layer.
    Returns PIL Image ready to use with NiceGUI's ui.image component.
    
    Args:
        app_state: Application state containing model information
        model_name: Name of the model to use
        question: Question containing the image path
        
    Returns:
        tuple: (PIL Image of visualization, predicted class name)
    """
    model_training = next((model_training for model_training in app_state.model_trainings 
                          if model_training.model_name == model_name), None)
    if model_training is None:
        raise ValueError(f"Model {model_name} not found")
    
    # Create the original model architecture based on model name
    if model_name == Models.ClassificationModel:
        original_model = get_classification_model()
    elif model_name == Models.ResNetModel:
        original_model = get_resnet_model()
    elif model_name == Models.EfficientNetModel:
        original_model = get_efficient_net_model()
    elif model_name == Models.VGGModel:
        original_model = get_vgg_model()
    else:
        raise ValueError(f"Unknown model architecture for {model_name}")
    
    # Load weights from the saved model
    original_model.load_state_dict(torch.jit.load(model_training.model_file_name).state_dict())
    original_model = original_model.to(app_state.device)
    original_model.eval()
    
    # Reenable the gradients for grad cam
    for param in original_model.parameters():
        param.requires_grad = True
    
    # Prepare the image
    transform = get_transformation()
    original_image = Image.open(question.image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(app_state.device)
    
    # Find target layer based on model architecture
    target_layer = None
    
    if model_name == Models.ClassificationModel:
        # For custom CNN, use the last conv layer in conv_block
        target_layer = original_model.conv_block[-3]  # Get the last Conv2d before the final MaxPool
    elif model_name == Models.ResNetModel:
        # For ResNet, use the last layer in layer4
        target_layer = original_model.layer4[-1].conv2
    elif model_name == Models.EfficientNetModel:
        # For EfficientNet, use the last conv layer in features
        # Navigate to the last MBConv block's last conv layer
        target_layer = original_model.features[-1][0]
        
    elif model_name == Models.VGGModel:
        # For VGG, use the last conv layer in features
        for module in original_model.features:
            if isinstance(module, nn.Conv2d):
                target_layer = module
        print(original_model.features)
        print(target_layer)
    if target_layer is None:
        raise ValueError(f"Could not find appropriate target layer for {model_name}")
    
    # Get activations and gradients
    activations = {}
    gradients = {}
    
    def forward_hook(module, input, output):
        activations['target'] = output
    
    def backward_hook(module, grad_input, grad_output):
        gradients['target'] = grad_output[0].clone()
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    output = original_model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    pred_class_name = app_state.class_names[pred_class]
    
    # Backward pass for the predicted class
    original_model.zero_grad()
    
    # For classification models with LogSoftmax
    if isinstance(output, torch.Tensor) and output.shape[1] == len(app_state.class_names):
        if model_name == Models.ClassificationModel:
            # If using LogSoftmax in the model (check the last layer)
            class_score = torch.exp(output[0, pred_class])  # Convert log probability back to probability
        else:
            class_score = output[0, pred_class]
    else:
        # Fallback
        class_score = output[0, pred_class]
    
    # Backward pass
    class_score.backward()
    
    # Remove the hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get the gradients and activations
    gradients = gradients['target']
    activations = activations['target']
    
    # Calculate weights based on global average pooling of gradients
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # Generate weighted activation map
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    
    # Apply ReLU to focus on features that have a positive influence
    cam = F.relu(cam)
    
    # Normalize the heatmap
    cam = cam - torch.min(cam)
    cam = cam / (torch.max(cam) + 1e-8)  # Add epsilon to avoid division by zero
    
    # Convert to numpy array
    cam = cam.detach().cpu().numpy()[0, 0]
    
    # Get original image dimensions
    original_width, original_height = original_image.size
    
    # Resize the heatmap to match original image size using PIL
    heatmap_array = (cam * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_array).resize((original_width, original_height), Image.BICUBIC)
    
    # Convert heatmap to colormap (similar to COLORMAP_JET in OpenCV)
    # Create a colormap similar to jet (going from blue to red)
    def create_colormap(value):
        # Map values from 0-1 to RGB colors simulating jet colormap
        if value < 0.25:
            r, g, b = 0, 4 * value, 1
        elif value < 0.5:
            r, g, b = 0, 1, 1 - 4 * (value - 0.25)
        elif value < 0.75:
            r, g, b = 4 * (value - 0.5), 1, 0
        else:
            r, g, b = 1, 1 - 4 * (value - 0.75), 0
        return int(r * 255), int(g * 255), int(b * 255)
    
    # Apply colormap to heatmap
    heatmap_colored = Image.new('RGB', heatmap_image.size)
    for y in range(heatmap_image.height):
        for x in range(heatmap_image.width):
            pixel_value = heatmap_image.getpixel((x, y)) / 255.0
            heatmap_colored.putpixel((x, y), create_colormap(pixel_value))
    
    # Convert original image to numpy array for blending
    original_array = np.array(original_image)
    heatmap_array = np.array(heatmap_colored)
    
    # Blend the heatmap with the original image (40% heatmap, 60% original)
    blended_array = (0.4 * heatmap_array + 0.6 * original_array).astype(np.uint8)
    
    # Convert back to PIL Image
    output_image = Image.fromarray(blended_array)
    
    return output_image, pred_class_name