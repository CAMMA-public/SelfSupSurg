import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchvision.transforms as transforms

class CholecT50():
    def __init__(self, dataset_dir, list_video, aug='original', split='train'):
        """ Args
                dataset_dir : common path to the dataset (excluding videos, output)
                list_video  : list video IDs, e.g:  ['VID01', 'VID02']
                aug         : data augumentation style
                split       : data split ['train', 'val', 'test']
            Call
                batch_size: int, 
                shuffle: True or False
            Return
                tuple ((image), (tool_label, verb_label, target_label, triplet_label))
        """
        transform = self.augmentation(aug)

        if split=='eval':
            video = list_video
            self.dataset = T50(img_dir = os.path.join(dataset_dir, 'data', video), 
                          triplet_file = os.path.join(dataset_dir, 'triplet', '{}.txt'.format(video)), 
                          tool_file = os.path.join(dataset_dir, 'instrument', '{}.txt'.format(video)),  
                          verb_file = os.path.join(dataset_dir, 'verb', '{}.txt'.format(video)),  
                          target_file = os.path.join(dataset_dir, 'target', '{}.txt'.format(video)), 
                          transform=transform)
        else:
            iterable_dataset = []
            for video in list_video:
                dataset = T50(img_dir = os.path.join(dataset_dir, 'data', video), 
                            triplet_file = os.path.join(dataset_dir, 'triplet', '{}.txt'.format(video)), 
                            tool_file = os.path.join(dataset_dir, 'instrument', '{}.txt'.format(video)),  
                            verb_file = os.path.join(dataset_dir, 'verb', '{}.txt'.format(video)),  
                            target_file = os.path.join(dataset_dir, 'target', '{}.txt'.format(video)), 
                            transform=transform)
                iterable_dataset.append(dataset)
            self.dataset = ConcatDataset(iterable_dataset)

    def augmentation(self, aug):
        def do_nothing(x):
            return x
        selector = {
            'original': do_nothing,
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90,expand=True),
            'sharpness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'autocontrast': transforms.RandomAutocontrast(p=0.5),
        }
        choice    = selector[aug]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize((256, 448)), choice, transforms.Resize((256, 448)), transforms.ToTensor(), normalize,])
        return transform
        
    def __call__(self, batch_size=2, num_workers=3, shuffle=False):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            
    
class T50(Dataset):
    def __init__(self, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None, target_transform=None):
        self.triplet_labels = np.loadtxt(triplet_file, dtype=np.int, delimiter=',',)
        self.tool_labels = np.loadtxt(tool_file, dtype=np.int, delimiter=',',)
        self.verb_labels = np.loadtxt(verb_file, dtype=np.int, delimiter=',',)
        self.target_labels = np.loadtxt(target_file, dtype=np.int, delimiter=',',)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.triplet_labels)
    
    def __getitem__(self, index):
        triplet_label = self.triplet_labels[index, 1:]
        tool_label = self.tool_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        img_path = os.path.join(self.img_dir, basename)
        # image    = io.imread(img_path)
        image    = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            triplet_label = self.target_transform(triplet_label)
        return image, (tool_label, verb_label, target_label, triplet_label)