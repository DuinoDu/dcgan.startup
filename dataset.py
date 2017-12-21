import torchvision.datasets as dset
import torchvision.transforms as transforms
import os

data_root = os.path.join(os.environ['HOME'], 'data')
if not os.path.exists(data_root):
    os.makedirs(data_root)

# Resize or Scale depending on pytorch version
def istorch_0_3():
    import torch
    version = torch.__version__[2]
    if int(version) >= 3:
        return True
    else:
        return False
    
if istorch_0_3():
    resize = resize
else:
    resize = transforms.Scale

def create_dataset(dataset_name, root=data_root, image_size=None):
    """Create dataset given dataset name. 

    args:
        dataset_name (str): dataset name

        image_size(int): output image size. Only for mnist, it's None

    returns:
        torch.util.data.dataset
    
    """
    path = os.path.join(root, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)

    if dataset_name == 'mnist':
        dataset = dset.MNIST(path, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(root=path, download=True,
                               transform=transforms.Compose([
                                   resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif dataset_name == 'cifar100':
        dataset = dset.CIFAR10(root=path, download=True,
                               transform=transforms.Compose([
                                   resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif dataset_name == 'lsun':
        dataset = dset.LSUN(db_path=path, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif dataset_name == 'fake':
        dataset = dset.FakeData(image_size=(3, image_size, image_size),
                                transform=transforms.ToTensor())
    else:
        raise "Unknown dataset name: ", dataset_name

    return dataset
