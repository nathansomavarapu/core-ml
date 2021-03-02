from torchvision import transforms
from typing import Tuple, Any

class BasicAugmentation:

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
                lambda x: x.convert('RGB'), # LABLEME contains black and white images
                transforms.RandomResizedCrop(
                    (224,224), 
                    scale=(0.8, 1.0)
                    ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(.4, .4, .4, .4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean, 
                    std=std
                    )
            ])
    
    def __call__(self, x):
        return self.transform(x)

class CifarVITTransform:

    def __init__(self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
    def __call__(self, x):
        return self.transform(x)

class BasicVITTransform:

    def __init__(self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
    def __call__(self, x):
        return self.transform(x)
