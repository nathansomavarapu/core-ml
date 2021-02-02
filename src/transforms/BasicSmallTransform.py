from torchvision import transforms

class BasicSmallTransform:

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform(x)
