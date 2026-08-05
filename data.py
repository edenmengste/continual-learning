import torch
from torchvision import datasets, transforms


def load_mnist():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])


    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )


    return train_dataset,test_dataset



class PermutedMNIST(torch.utils.data.Dataset):

    def __init__(self,dataset,seed=42):

        self.dataset = dataset

        torch.manual_seed(seed)

        self.permutation = torch.randperm(784)



    def __len__(self):

        return len(self.dataset)



    def __getitem__(self,index):

        image,label = self.dataset[index]


        image = image.view(-1)


        image = image[self.permutation]


        image=image.view(
            1,
            28,
            28
        )


        return image,label