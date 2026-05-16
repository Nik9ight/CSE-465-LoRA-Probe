import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from dataset import get_dataset, get_dataloader, get_subset
from models import FeatureExtractor
from peft import PeftModel

def extract_features(model_name, use_lora=False, lora_path=None, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Extracting features for {model_name} (LoRA={use_lora})...")
    
    dataset = get_dataset(split='test', download=True)
    # dataset = get_subset(dataset, num_classes=10) # Match training subset if used
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    
    if use_lora and lora_path:
        print(f"Loading LoRA from {lora_path}")
        # Load base model first (without LoRA)
        fe = FeatureExtractor(model_name, use_lora=False, num_classes=37)
        # Load the saved LoRA adapters
        fe.model = PeftModel.from_pretrained(fe.model, lora_path)
        fe.model = fe.model.to(device)
        # Note: Skip merge_and_unload() for models with depthwise convolutions (groups > 1)
        # like MobileNetV2, as merging is not supported. Use the model with adapters directly.
        fe.model.eval()
        print(f"Model type with LoRA adapters: {type(fe.model)}")
    else:
        # Use base model without LoRA
        fe = FeatureExtractor(model_name, use_lora=False, num_classes=37)
    
    features = []
    labels = []
    predictions = []  # Track predictions to check if model is actually different
    images_list = [] # Store paths or indices if needed, but dataset has them
    
    fe.model.eval()  # Ensure model is in eval mode
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader)):
            images = images.to(device)
            feats = fe.get_embedding(images)
            
            # Also get predictions to verify the model is actually using LoRA weights
            logits = fe.model(images)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            
            # Debug: print first batch stats
            if batch_idx == 0:
                print(f"First batch feature stats - mean: {feats.mean().item():.4f}, std: {feats.std().item():.4f}, min: {feats.min().item():.4f}, max: {feats.max().item():.4f}")
                print(f"First batch predictions: {preds.cpu().numpy()}")
                print(f"First batch ground truth: {targets.numpy()}")
            
            # Normalize features for better comparison
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())
            
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    predictions = np.array(predictions)
    
    # Calculate classification accuracy on test set
    accuracy = np.mean(predictions == labels)
    print(f"Classification accuracy on test set: {accuracy:.4f}")
    
    return features, labels, dataset

def evaluate_knn(features, labels, dataset, k=10, output_dir='./results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # Compute Precision@K
    precisions = []
    for i in range(len(labels)):
        query_label = labels[i]
        neighbor_indices = indices[i][1:] # Exclude self
        neighbor_labels = labels[neighbor_indices]
        precision = np.sum(neighbor_labels == query_label) / k
        precisions.append(precision)
        
    avg_precision = np.mean(precisions)
    print(f"Average Precision@{k}: {avg_precision:.4f}")
    
    return avg_precision, indices

def visualize_results(dataset, indices, labels, model_name, mode, num_queries=5, output_dir='./results'):
    # Select random queries
    query_indices = np.random.choice(len(dataset), num_queries, replace=False)
    
    for i, query_idx in enumerate(query_indices):
        fig, axes = plt.subplots(1, 11, figsize=(20, 3))
        
        # Query image
        img, label = dataset[query_idx]
        # Unnormalize for display
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[0].imshow(img)
        axes[0].set_title(f"Query\nClass: {label}")
        axes[0].axis('off')
        
        # Neighbors
        neighbor_indices = indices[query_idx][1:]
        for j, n_idx in enumerate(neighbor_indices):
            n_img, n_label = dataset[n_idx]
            n_img = n_img.permute(1, 2, 0).numpy()
            n_img = n_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            n_img = np.clip(n_img, 0, 1)
            
            axes[j+1].imshow(n_img)
            axes[j+1].set_title(f"N{j+1}\nClass: {n_label}")
            axes[j+1].axis('off')
            
            # Highlight correct class
            if n_label == label:
                for spine in axes[j+1].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2)
            else:
                for spine in axes[j+1].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
                    
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_{mode}_query_{i}.png")
        plt.close()

if __name__ == "__main__":
    # Test evaluation
    feats, lbls, ds = extract_features('resnet101', batch_size=4)
    evaluate_knn(feats, lbls, ds)
