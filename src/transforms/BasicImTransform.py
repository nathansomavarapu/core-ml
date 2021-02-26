from torchvision import transforms

class BasicImTransform:

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            lambda x: x.convert('RGB'), # Some datasets contain B&W images and only color images are supported with imagenet.
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])
    
    def __call__(self, x):
        return self.transform(x)
