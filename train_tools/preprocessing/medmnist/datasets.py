from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from medmnist import BloodMNIST, DermaMNIST, OrganCMNIST, OrganSMNIST, OCTMNIST, TissueMNIST


class MedMNISTDataset(Dataset):
    """MedMNIST dataset class"""

    def __init__(self, root='/media/zoe/ssd/', dataset='BloodMNIST', transform=None, dataidxs=None, mode='tr'):
        """Initialize MedMNISTDataset."""
        if mode == 'tr':
            extra = 'train'
        elif mode == 'val':
            extra = 'val'
        else:
            extra = 'test'

        self.dataset = dataset

        try:
            #print(root + '/' + extra + '_images.npy')
            x = np.load(root + '/' + extra + '_images.npy')
            y = np.load(root + '/' + extra + '_labels.npy')

        except FileNotFoundError:

            if self.dataset == 'BloodMNIST':
                img_dir = ('/').join((root.split('/'))[:-1])
                data = BloodMNIST(split=extra, download=True, as_rgb=True, root=img_dir)
                x = data.imgs
                y = data.labels
            elif self.dataset == 'DermaMNIST':
                img_dir = ('/').join((root.split('/'))[:-1])
                data = DermaMNIST(split=extra, download=True, as_rgb=True, root=img_dir)
                x = data.imgs
                y = data.labels
            elif self.dataset == 'OrganCMNIST':
                img_dir = ('/').join((root.split('/'))[:-1])
                data = OrganCMNIST(split=extra, download=True, as_rgb=False, root=img_dir)
                x = data.imgs
                y = data.labels
            elif self.dataset == 'OCTMNIST':
                img_dir = ('/').join((root.split('/'))[:-1])
                data = OCTMNIST(split=extra, download=True, as_rgb=False, root=img_dir)
                x = data.imgs
                y = data.labels
            elif self.dataset == 'OrganSMNIST':
                img_dir = ('/').join((root.split('/'))[:-1])
                data = OrganSMNIST(split=extra, download=True, as_rgb=False, root=img_dir)
                x = data.imgs
                y = data.labels
            elif self.dataset == 'TissueMNIST':
                img_dir = ('/').join((root.split('/'))[:-1])
                data = TissueMNIST(split=extra, download=True, as_rgb=False, root=img_dir)
                x = data.imgs
                y = data.labels


        if dataidxs is not None:
            x = x[dataidxs]
            y = y[dataidxs]

        self.x, self.targets = x, y.squeeze()
        self.transform = transform

    def __getitem__(self, index: int):
        """Return an item by the index."""
        img = self.x[index]
        label = self.targets[index]
        img = Image.fromarray(img)
        if 'Pneumonia' in self.dataset:
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label, index

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.x)
