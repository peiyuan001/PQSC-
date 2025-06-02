import torch
import torchvision
from pathlib import Path
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import random

class ShuffledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        shuffled_index = self.indices[index]
        return self.dataset[shuffled_index]


def get_dataset(task: str, cfg, shuffle_train=True, shuffle_test=False, return_dataset=False):

    show_num = 3

    if task == 'cifar10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        train_dataset = torchvision.datasets.CIFAR10('/data/datasets', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10('/data/datasets', train=False, transform=transforms, download=True)
        train_dataset = ShuffledDataset(train_dataset)
        show_dataset = Subset(test_dataset, range(0, show_num))

    else:
        print("> Unknown dataset. Terminating")
        exit()

    print(f"> Train dataset size: {len(train_dataset)}")
    print(f"> Test dataset size: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.nb_workers, shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.nb_workers, shuffle=shuffle_test)
    show_loader = torch.utils.data.DataLoader(show_dataset, batch_size=cfg.batch_size, num_workers=cfg.nb_workers, shuffle=shuffle_test)
    if return_dataset:
        return (train_loader, test_loader, show_loader), (train_dataset, test_dataset, show_dataset)

    return train_loader, test_loader, show_loader
