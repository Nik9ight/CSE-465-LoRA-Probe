import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from dataset import get_dataset, get_dataloader, get_subset
from models import FeatureExtractor

def train_model(model_name, epochs=5, batch_size=32, lr=1e-3, lora_r=8, save_path='./models'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Training {model_name} on {device} with LoRA r={lora_r}")
    
    # Load dataset
    dataset = get_dataset(split='trainval', download=True)
    # Use a subset for faster demonstration if needed, but let's try full dataset first
    # dataset = get_subset(dataset, num_classes=10) 
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
    
    # Load model
    fe = FeatureExtractor(model_name, use_lora=True, lora_r=lora_r, num_classes=37)
    model = fe.model
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing for better generalization
    # Use lower learning rate to prevent catastrophic forgetting
    optimizer = optim.AdamW(model.parameters(), lr=lr*0.5, weight_decay=0.1)  # Lower LR, higher weight decay
    
    # Add learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*0.5, steps_per_epoch=len(loader), epochs=epochs)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle different output types (GoogLeNet returns tuple in train mode)
            if model_name == 'googlenet' and hasattr(outputs, 'logits'):
                outputs = outputs.logits
            elif isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
            scheduler.step()  # OneCycleLR steps per batch
        
        print(f"Epoch {epoch+1} - Loss: {running_loss/total:.6f}, Accuracy: {correct/total:.4f}")
            
    # Save model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model.save_pretrained(os.path.join(save_path, f"{model_name}_lora"))
    print(f"Model saved to {save_path}/{model_name}_lora")

if __name__ == "__main__":
    train_model('resnet101', epochs=1)
