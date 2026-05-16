import argparse
import os
from train import train_model
from evaluate import extract_features, evaluate_knn, visualize_results

def main():
    parser = argparse.ArgumentParser(description="CNN Feature Analysis with LoRA")
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'train', 'evaluate_lora'], help='Mode to run')
    parser.add_argument('--model', type=str, default='resnet101', choices=['resnet101', 'googlenet', 'vit_b_16', 'mobilenet_v2'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA rank')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save/load models')
    
    args = parser.parse_args()
    
    if args.mode == 'baseline':
        print(f"Running baseline analysis for {args.model}...")
        features, labels, dataset = extract_features(args.model, use_lora=False, batch_size=args.batch_size)
        avg_prec, indices = evaluate_knn(features, labels, dataset, output_dir=args.output_dir)
        visualize_results(dataset, indices, labels, args.model, 'baseline', output_dir=args.output_dir)
        
    elif args.mode == 'train':
        print(f"Fine-tuning {args.model} with LoRA...")
        train_model(args.model, epochs=args.epochs, batch_size=args.batch_size, lr=1e-3, lora_r=args.lora_r, save_path=args.model_dir)
        
    elif args.mode == 'evaluate_lora':
        print(f"Evaluating LoRA-tuned {args.model}...")
        lora_path = os.path.join(args.model_dir, f"{args.model}_lora")
        features, labels, dataset = extract_features(args.model, use_lora=True, lora_path=lora_path, batch_size=args.batch_size)
        avg_prec, indices = evaluate_knn(features, labels, dataset, output_dir=args.output_dir)
        visualize_results(dataset, indices, labels, args.model, 'lora', output_dir=args.output_dir)

if __name__ == "__main__":
    main()
