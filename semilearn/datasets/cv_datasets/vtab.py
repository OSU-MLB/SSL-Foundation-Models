import os
import copy
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from torchvision.datasets import folder as dataset_parser
from torchvision.datasets import ImageFolder
from timm.data.transforms import str_to_interp_mode
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data

_SPLIT2IMGLIST = {
    # Original dataset
    'train': 'train.list',
    'val': 'val.list',
    'trainval': 'trainval.list',
    'test': 'test.list',
    # # TODO: 1k subset
    # 'train800': 'train800.list',
    # 'val200': 'val200.list',
    # 'train800val200': 'train800val200.list',
}

class VTAB(ImageFolder, BasicDataset):

    def __init__(self, alg, root, split, 
                idx_list=None,
                is_ulb=False,
                transform=None, 
                medium_transform=None, 
                strong_transform=None):
        """see comments at the beginning of the script"""
        img_root = os.path.join(root, 'images')
        # get absolute path
        super(VTAB, self).__init__(img_root, transform=transform)
        self.root = root

        # Check if split is valid
        assert split in _SPLIT2IMGLIST, f"split {split} is not supported. "
        
        # Set attributes
        self.split = split
        self.imagelist_file = None if split is None else _SPLIT2IMGLIST[split]
        self.samples = None
        self.targets = None
        self.idx_list = None
        self._samples = None
        self._targets = None
        self._idx_list = None
        self.num_classes = None
        self.init_samples(idx_list)


        ### 10/07: mro is broken because torch.utils.data.Dataset doesn't have __init__ method, thus have to call it explicitly
        BasicDataset.__init__(self, alg, 
                              self.samples, 
                              self.targets, 
                              self.num_classes, 
                              transform, 
                              is_ulb, 
                              medium_transform, 
                              strong_transform,
                              idx_list=self.idx_list)

    def set_targets(self, targets):
        assert len(targets) == len(self.samples), f"Length of targets {len(targets)} is not equal to length of samples {len(self.samples)}. "
        self.targets = targets

    # def set_idx_list(self, idx_list):
    #     self.idx_list = idx_list

    def update_idx_list(self, new_idx, y_true=None, y_pred=None):
        self.idx_list = self._idx_list[new_idx]
        self.samples = self._samples[self.idx_list]
        self.targets = self._targets[self.idx_list]

    def init_samples(self, idx_list):
        # Load samples and targets
        lines = np.loadtxt(os.path.join(self.root, self.imagelist_file), 
                           dtype={'names': ('img_path', 'target'), 
                                  'formats': ('U100', 'int64')})
        samples = lines['img_path']
        samples = [os.path.join(self.root, s) for s in samples]
        targets = (lines['target'])
        samples = np.array(samples)
        targets = np.array(targets)

        # Set attributes
        self.num_classes = len(set(targets))
        if idx_list is not None:
            self.idx_list = np.array(idx_list)
        else:
            self.idx_list = np.arange(len(samples))
        
        self.samples = samples[self.idx_list]
        self.targets = targets[self.idx_list]
        self._samples = copy.deepcopy(samples)
        self._targets = copy.deepcopy(targets)
        self._idx_list = copy.deepcopy(self.idx_list)
    
    def __sample__(self, index):
        # idx = self.idx_list[index]
        # path, target =  self.samples[idx], self.targets[idx]
        path, target =  self.samples[index], self.targets[index]
        img = self.loader(path)
        return img, target

    def __getitem__(self, index):
        return BasicDataset.__getitem__(self, index)

    def __len__(self):
        return len(self.idx_list)


transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]


def _get_vtab_transforms(img_size, model_name):

    weak_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size), interpolation=str_to_interp_mode('bicubic')),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(transform_mean, transform_std)
    ])

    medium_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size), interpolation=str_to_interp_mode('bicubic')),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(transform_mean, transform_std)
    ])

    strong_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size), interpolation=str_to_interp_mode('bicubic')),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(transform_mean, transform_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=str_to_interp_mode('bicubic')),
        transforms.ToTensor(),
        transforms.Normalize(transform_mean, transform_std)
    ])

    return weak_transform, medium_transform, strong_transform, val_transform


def get_vtab(args, alg, dset_name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):

    # Sanity check
    assert args.dataset == dset_name, f"Dataset {dset_name} is not equal to args.dataset {args.dataset}. "
    assert args.num_labels == num_labels, f"Num of labels {num_labels} is not equal to args.num_labels {args.num_labels}. "

    # Unpack arguments
    ulb_num_labels=args.ulb_num_labels
    lb_imbalance_ratio=args.lb_imb_ratio
    ulb_imbalance_ratio=args.ulb_imb_ratio
    model_name = args.net
    if hasattr(args, 'train_split'):
        train_split = args.train_split
    else:
        train_split = None
    crop_ratio = args.crop_ratio
    print(f"crop_ratio: {crop_ratio} is not used in VTAB dataset. ")
    img_size = args.img_size
    assert img_size == 224, f"Image size {img_size} is not equal to 224. "

    # Get transforms
    weak_transform, medium_transform, strong_transform, val_transform = _get_vtab_transforms(img_size, model_name)

    # Load dataset
    data_dir = os.path.join(data_dir, 'vtab', dset_name.lower())

    if train_split is not None:
        # Get samples and targets
        _base_dataset = VTAB(alg, data_dir, split=train_split)
        train_samples = _base_dataset.samples
        train_targets = _base_dataset.targets
        train_idx = _base_dataset.idx_list
    
        assert len(train_targets) == len(train_idx), f"{dset_name} dataset has error: len(targets) != len(ids). "

        # Split data into labeled and unlabeled
        if alg in ['fullysupervised']:
            train_labeled_idxs = train_idx
            train_unlabeled_idxs = []
        else:
            train_labeled_idxs, _, _, train_unlabeled_idxs, _, _, ulb_lb_mask, = \
                split_ssl_data(args, 
                            data=train_samples, 
                            targets=train_targets, 
                            num_classes=num_classes,
                            lb_num_labels=num_labels,
                            ulb_num_labels=ulb_num_labels, 
                            lb_imbalance_ratio=lb_imbalance_ratio, ulb_imbalance_ratio=ulb_imbalance_ratio, include_lb_to_ulb=include_lb_to_ulb,
                            data_dir=data_dir, 
                            save_format='list',
                            return_idxs=True,
                            save_appendix=f"{train_split}_")
        print(f"Num of labeled: {len(train_labeled_idxs)}, Num of unlabeled: {len(train_unlabeled_idxs)}")

        print(f"train_labeled_idxs: {train_labeled_idxs}")
        print(f"train_labels: {train_targets[train_labeled_idxs]}")

        assert hasattr(args, 'train_aug'), f"train_aug is not in args. "
        if args.train_aug == 'none':
            print("Using no augmentation. ")
            train_transform = val_transform
        elif args.train_aug == 'weak':
            print("Using weak augmentation. ")
            train_transform = weak_transform
        elif args.train_aug == 'strong':
            print("Using strong augmentation. ")
            train_transform = strong_transform
        else:
            raise ValueError(f"train_aug {args.train_aug} is not supported. ")

        # Prepare datasets
        train_labeled_dataset = VTAB(alg, 
                                    data_dir, 
                                    split=train_split,
                                    is_ulb=False,
                                    idx_list=train_labeled_idxs,
                                    transform=train_transform,
                                    strong_transform=strong_transform)

        train_unlabeled_dataset = VTAB(alg, 
                                    data_dir, 
                                    split=train_split,
                                    is_ulb=True, 
                                    idx_list=train_unlabeled_idxs, 
                                    transform=train_transform, 
                                    medium_transform=medium_transform,
                                    strong_transform=strong_transform)
        

        train_labeled_dataset.val_transform = val_transform
        train_unlabeled_dataset.val_transform = val_transform

    else:
        train_labeled_dataset = None
        train_unlabeled_dataset = None
    
    val_dataset = VTAB(alg, 
                      data_dir, 
                      split='val', 
                      transform=val_transform)
    test_dataset = VTAB(alg,
                        data_dir,
                        split='test',
                        transform=val_transform)

    print(f"Num of labeled: {'None' if train_labeled_dataset is None else len(train_labeled_dataset)}, "
          f"Num of unlabeled: {'None' if train_unlabeled_dataset is None else len(train_unlabeled_dataset)}, "
          f"Num of val: {len(val_dataset)}, "
          f"Num of test: {len(test_dataset)}")
    
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, ulb_lb_mask
