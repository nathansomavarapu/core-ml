from torchvision import transforms

class BasicSmallTransform:

    def __init__(self, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
        self.transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])
    
    def __call__(self, x):
        return self.transform(x)
