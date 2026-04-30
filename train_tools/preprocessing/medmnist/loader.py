from torchvision.transforms import transforms

from train_tools.preprocessing.medmnist.datasets import MedMNISTDataset
import torch
import torch.utils.data as data

def get_all_targets_medmnist(root, train='tr', dataset_label='BloodMNIST'):
    print(root)
    dataset = MedMNISTDataset(root=root, mode=train, dataset=dataset_label)
    all_targets = dataset.targets.squeeze()
    return all_targets

def get_transforms(dataset):
    if dataset == 'BloodMNIST':
        mean = [0.5] #[0.7943478411516803, 0.6596590061729185, 0.6961925053489914]
        std = [0.5]
        #[0.21563032622411277, 0.24160339316387083, 0.11788897316984681]
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    elif dataset == 'DermaMNIST':
        mean = [0.5]
        std = [0.5]
        # mean = [0.7631127033373645, 0.5380881751873002, 0.5613881416702554]
        # std = [0.13660512607124103, 0.15425382682707006, 0.16921601147359838]
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    elif dataset == 'OrganCMNIST':
        mean = [0.5]
        std = [0.5]
        # mean = [0.7631127033373645, 0.5380881751873002, 0.5613881416702554]
        # std = [0.13660512607124103, 0.15425382682707006, 0.16921601147359838]
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    elif dataset == 'OrganSMNIST':
        mean = [0.5]
        std = [0.5]
        # mean = [0.7631127033373645, 0.5380881751873002, 0.5613881416702554]
        # std = [0.13660512607124103, 0.15425382682707006, 0.16921601147359838]
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    elif dataset == 'OCTMNIST':
        mean = [0.5]
        std = [0.5]
        # mean = [0.7631127033373645, 0.5380881751873002, 0.5613881416702554]
        # std = [0.13660512607124103, 0.15425382682707006, 0.16921601147359838]
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    elif dataset == 'TissueMNIST':
        mean = [0.5]
        std = [0.5]
        # mean = [0.7631127033373645, 0.5380881751873002, 0.5613881416702554]
        # std = [0.13660512607124103, 0.15425382682707006, 0.16921601147359838]
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    elif (dataset == 'PneumoniaMNIST') or (dataset == 'Kermany') or (dataset == 'CXR8'):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    else:
        mean = .1706
        std = .2112
        data_transforms = {
            'train': transforms.Compose([
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    return data_transforms


def get_dataloader_medmnist(root, mode='tr', batch_size=50, dataidxs=None, dataset_label='BloodMNIST'):

    transforms_total = get_transforms(dataset=dataset_label)
    train_transform = transforms_total['train']

    if mode == 'tr':
        dataset = MedMNISTDataset(
            root, mode=mode, transform=train_transform, dataset=dataset_label, dataidxs=dataidxs
        )
        # Create overall dataloader
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

    else:
        dataset = MedMNISTDataset(
            root, mode=mode, transform=transforms_total['test'], dataset=dataset_label, dataidxs=dataidxs
        )
        # Create overall dataloader
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

    return dataloader
