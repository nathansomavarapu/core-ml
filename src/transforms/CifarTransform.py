from torchvision import transforms

class CifarTransform:

    def __init__(self, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, x):
        return self.transform(x)
