import torch
import torch.nn as nn
import torchvision.models as models
from peft import get_peft_model, LoraConfig, TaskType

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet101', use_lora=False, lora_r=8, num_classes=37):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
             self.device = torch.device("mps")

        self.model = self._load_model(model_name, num_classes)
        
        if use_lora:
            self.apply_lora(r=lora_r)

        self.model.to(self.device)
        self.model.eval()
        # Ensure all modules are in eval mode
        for module in self.model.modules():
            if hasattr(module, 'training'):
                module.training = False

    def _load_model(self, model_name, num_classes):
        if model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            # Replace head for fine-tuning if needed, but for feature extraction we usually remove it.
            # However, for LoRA training we need a head to train on.
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'googlenet':
            model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'vit_b_16':
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported")
        return model

    def apply_lora(self, r=8):
        if self.model_name == 'vit_b_16':
             config = LoraConfig(
                r=r,
                lora_alpha=r*2,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["heads"],
            )
        elif self.model_name == 'resnet101':
             # Target all 3x3 convolutions (conv2) in the bottlenecks
             config = LoraConfig(
                r=r,
                lora_alpha=r*2,
                target_modules=r".*\.conv2", 
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["fc"],
            )
        elif self.model_name == 'googlenet':
             # Target Conv2d layers inside BasicConv2d modules
             # The pattern targets Conv2d modules that are children of BasicConv2d (named 'conv')
             config = LoraConfig(
                r=r,
                lora_alpha=r*2,
                target_modules=r".*\.conv$", 
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["fc"],
            )
        elif self.model_name == 'mobilenet_v2':
             # Target only 1x1 pointwise Conv2d layers in MobileNetV2's InvertedResidual blocks
             # conv.0.0 = expand Conv2d (1x1, groups=1)
             # Only target expand convolutions to avoid matching BatchNorm layers
             config = LoraConfig(
                r=r,
                lora_alpha=r*2,
                target_modules=r"features\.\d+\.conv\.0\.0$", 
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
            )

        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

    def get_embedding(self, x):
        # We need to extract features before the classification head.
        # We can use a hook or forward up to penultimate.
        
        features = []
        def hook(module, input, output):
            features.append(output.flatten(1))
            
        handle = None
        # Handle both PEFT wrapped models and base models
        if hasattr(self.model, 'base_model'):
            base = self.model.base_model.model if hasattr(self.model.base_model, 'model') else self.model.base_model
        else:
            base = self.model

        if self.model_name == 'resnet101':
            handle = base.avgpool.register_forward_hook(hook)
            
        elif self.model_name == 'googlenet':
            handle = base.dropout.register_forward_hook(hook)
            
        elif self.model_name == 'vit_b_16':
            handle = base.encoder.ln.register_forward_hook(hook)

        elif self.model_name == 'mobilenet_v2':
            # MobileNetV2: features -> adaptive_avg_pool2d -> classifier
            # We want the output of the pooling layer, which is not explicitly a named module in standard implementation
            # usually it's nn.functional.adaptive_avg_pool2d called in forward.
            # However, torchvision implementation has `classifier` as a Sequential.
            # The forward is: x = features(x); x = nn.functional.adaptive_avg_pool2d(x, (1, 1)); x = torch.flatten(x, 1); x = classifier(x)
            # So we can't easily hook onto the pool because it's functional.
            # We can hook onto the last layer of `features` and do the pooling ourselves, OR
            # we can hook onto the input of the classifier?
            # Hooking input of classifier[0] (Dropout) seems safest.
            # Handle PEFT's ModulesToSaveWrapper - unwrap to get the actual Sequential module
            classifier = base.classifier
            # PEFT wraps modules_to_save in ModulesToSaveWrapper with a 'modules_to_save' dict
            if hasattr(classifier, 'modules_to_save'):
                # Get the default adapter's module
                classifier = list(classifier.modules_to_save.values())[0]
            elif hasattr(classifier, 'original_module'):
                classifier = classifier.original_module
            handle = classifier[0].register_forward_hook(lambda m, i, o: features.append(i[0].flatten(1)))

        with torch.no_grad():
            self.model(x)
        
        handle.remove()
        return features[0]

if __name__ == "__main__":
    print("Testing model loader...")
    # Test ResNet
    fe = FeatureExtractor('resnet101', use_lora=True)
    dummy = torch.randn(1, 3, 224, 224).to(fe.device)
    emb = fe.get_embedding(dummy)
    print(f"ResNet embedding shape: {emb.shape}")
    
    # Test ViT
    fe_vit = FeatureExtractor('vit_b_16', use_lora=False)
    dummy = torch.randn(1, 3, 224, 224).to(fe_vit.device)
    emb = fe_vit.get_embedding(dummy)
    print(f"ViT embedding shape: {emb.shape}")
