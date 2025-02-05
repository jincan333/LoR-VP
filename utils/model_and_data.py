import os
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset, RandomSampler, DistributedSampler, SequentialSampler, Sampler
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, GTSRB, Food101, SUN397, EuroSAT, UCF101, DTD, OxfordIIITPet, MNIST, ImageFolder
from torchvision import  transforms
import torch
import numpy as np
from PIL import Image
import lmdb
import pickle
import six
import json
from collections import OrderedDict
from utils.const import IMAGENETNORMALIZE, CIFAR100NORMALIZE, CIFAR10NORMALIZE
from models.vits import VisionTransformer, CONFIGS
import timm
import logging
from transformers import ViTModel
from utils.label_mapping import label_mapping_base
from functools import partial
import io


logger = logging.getLogger(__name__)

def get_model(net, args):
    torch.hub.set_dir('./cache')
    # network
    if net == "resnet18":
        if args.pretrain_path:
            from torchvision.models import resnet18, ResNet18_Weights
            network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            network = resnet18(pretrained = False)
    elif net == "resnet50":
        if args.pretrain_path:
            from torchvision.models import resnet50, ResNet50_Weights
            network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            network = resnet50(pretrained = False)
    elif net == "resnet50_21k":
        network = timm.create_model('resnet50', pretrained=False, num_classes=11221)
        if args.pretrain_path:
            network = load_model_weights(network, args.pretrain_path)
    elif net == "instagram":
        if args.pretrain_path:
            from torch import hub
            network = hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        else:
            from torchvision.models import resnext101_32x8d
            network = resnext101_32x8d(pretrained=False)
    elif net == 'mobilenet':
        if args.pretrain_path:
            from torchvision.models import mobilenet_v3
            network = mobilenet_v3(pretrained=True)
        else:
            network = mobilenet_v3(pretrained=False)
    elif net == 'vit_b_16_21k':
        network = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=True)
    elif net == 'vit_b_16_1k':
        network = timm.create_model("hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True)
    elif net == 'vit_b_32_1k':
        network = timm.create_model("hf_hub:timm/vit_base_patch32_224.augreg_in21k_ft_in1k", pretrained=True)
    elif net == 'swin_b':
        network = timm.create_model('swin_base_patch4_window7_224.ms_in22k', pretrained=True)
    elif 'vit' in net.lower() or 'test' in net.lower():
        # network = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        config = CONFIGS[net]
        network = VisionTransformer(config, args.output_size, num_classes=21843, head_init=args.head_init)
        if args.pretrain_path:
            if args.pretrain_path.endswith('.bin'):
                checkpoint = torch.load(args.pretrain_path, map_location=args.device)
                network.load_state_dict(checkpoint)
            else:
                weights = np.load(args.pretrain_path)
                print(weights.keys())
                network.load_from(weights)
        logger.info("Model Config:\n{}".format(config))
    else:
        raise NotImplementedError(f"{net} is not supported")
    # print('Network Architecture:', network)

    return network


def get_specified_model(network, args):
    ckpt = torch.load(args.specified_path, map_location='cpu')
    if network.visual_prompt and ckpt['visual_prompt']:
        network.visual_prompt.load_state_dict(ckpt['visual_prompt'])
    if ckpt['mapping_sequence']:
        network.mapping_sequence = ckpt['mapping_sequence']
        network.label_mapping = partial(label_mapping_base, mapping_sequence=network.mapping_sequence)
    if ckpt['network'] is not None:
        network.load_state_dict(ckpt['network'])
    
    return network




def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


# Imagenet Transform
def image_transform(args, transform_type):
    if '21k' in args.network.lower():
        normalize = None
    elif 'vit' in args.network.lower():
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=IMAGENETNORMALIZE['mean'], std=IMAGENETNORMALIZE['std'])
    if transform_type == 'vp':
        if args.randomcrop:
            print('Using randomcrop\n')
            train_transform = transforms.Compose([
                transforms.Resize((int(args.input_size*9/8), int(args.input_size*9/8))),
                transforms.RandomCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
                transforms.ToTensor(),
            ])
    elif transform_type == 'ff':
        train_transform = transforms.Compose([
            transforms.Resize((int(args.input_size*9/8), int(args.input_size*9/8))),
            transforms.RandomCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            normalize
        ])
    return train_transform, test_transform, normalize


def get_torch_dataset(args, transform_type):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    data_path = os.path.join(args.datadir, args.dataset)
    dataset = args.dataset
    train_transform, test_transform, normalize = image_transform(args, transform_type)

    if dataset == "cifar10":
        train_set = CIFAR10(os.path.join(args.datadir, 'cifar10'), train=True, transform=train_transform, download=True)
        test_set = CIFAR10(os.path.join(args.datadir, 'cifar10'), train=False, transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == "cifar100":
        train_set = CIFAR100(os.path.join(args.datadir, 'cifar100'), train=True, transform=train_transform, download=True)
        test_set = CIFAR100(os.path.join(args.datadir, 'cifar100'), train=False, transform=test_transform, download=True)
        class_cnt = 100

    elif dataset == "svhn":
        train_set = SVHN(data_path, split = 'train', transform=train_transform, download=True)
        test_set = SVHN(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == "gtsrb":
        full_data = GTSRB(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(GTSRB(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        test_set = GTSRB(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 43

    elif dataset == 'food101':
        train_set = Food101(data_path, split = 'train', transform=train_transform, download=True)
        test_set = Food101(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 101

    elif dataset == 'sun397':
        img_dir = os.path.join(data_path, 'SUN397')
        train_partition_file = os.path.join(data_path, 'Partitions/Training_01.txt')
        test_partition_file = os.path.join(data_path, 'Partitions/Testing_01.txt')
        target_file = os.path.join(data_path, 'Partitions/ClassName.txt')
        full_data = SUN397Dataset(img_dir = img_dir, partition_file = train_partition_file,  target_file=target_file)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(SUN397Dataset(img_dir = img_dir, partition_file = train_partition_file, target_file=target_file, transform=train_transform), train_indices)
        test_set = SUN397Dataset(img_dir = img_dir, partition_file = test_partition_file, target_file=target_file, transform=test_transform)
        class_cnt = 397

    elif dataset == 'eurosat':
        full_data = EuroSAT(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(EuroSAT(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        test_set = EuroSAT(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == 'ucf101':
        annotation_path = os.path.join(data_path, 'ucfTrainTestlist')
        data_path = os.path.join(data_path, 'UCF-101')
        full_data = UCF101(root = data_path,  annotation_path=annotation_path, frames_per_clip=1, fold=1, train = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(UCF101(data_path, train = True, annotation_path=annotation_path, frames_per_clip=1, fold=1, transform=train_transform), train_indices)
        test_set = UCF101(data_path, train = False, annotation_path=annotation_path, frames_per_clip=1, fold=1, transform=test_transform)
        class_cnt = 101
    
    elif dataset == 'stanfordcars':
        # 128
        data_path = os.path.join(data_path, 'car_data/car_data')
        train_set = ImageFolder(data_path+'/train/', transform=train_transform)
        test_set = ImageFolder(data_path+'/test/', transform=test_transform)
        class_cnt = 196
    
    elif dataset == 'flowers102':
        # 128
        train_set = ConcatDataset([COOPLMDBDataset(root = data_path, split="train", transform = train_transform), \
                                   COOPLMDBDataset(root = data_path, split="val", transform = train_transform)])
        test_set = COOPLMDBDataset(root = data_path, split="test", transform = test_transform)
        class_cnt = 102
    
    elif dataset == 'dtd':
        # 64
        train_set = ConcatDataset([DTD(root = data_path, split = 'train', transform=train_transform, download = True), \
                                DTD(root = data_path, split = 'val', transform=train_transform, download = True)])
        test_set = DTD(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 47

    elif dataset == 'oxfordpets':
        # 64
        train_set = OxfordIIITPet(data_path, split = 'trainval', transform=train_transform, download=True)
        test_set = OxfordIIITPet(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 37
    
    elif dataset == 'mnist':
        train_set = MNIST(data_path, train = True, transform=train_transform, download=True)
        test_set = MNIST(data_path, train = False, transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == 'imagenet':
        # 400*350  102
        if args.mode == 'train':
            train_set = ImageFolder(os.path.join(args.datadir, 'ImageNet/ILSVRC2012_train/data'), transform=train_transform)
        else:
            train_set = ImageFolder(os.path.join(args.datadir, 'ImageNet/ILSVRC2012_validation/data'), transform=train_transform)
        test_set = ImageFolder(os.path.join(args.datadir, 'ImageNet/ILSVRC2012_validation/data'), transform=test_transform)
        class_cnt = 1000
    
    elif dataset == 'tiny_imagenet':
        # 64*64*3  256
        train_set = ImageFolder(root=os.path.join(args.datadir, 'tiny-imagenet-200/train'), transform=train_transform)
        test_set = TinyImageNet(os.path.join(args.datadir, 'tiny-imagenet-200/val/images'), os.path.join(args.datadir, 'tiny-imagenet-200/val/val_annotations.txt'), 
                                os.path.join(args.datadir, 'tiny-imagenet-200/wnids.txt'), transform=test_transform)
        class_cnt = 200
    elif args.dataset == 'imagenet-r':
        normalize = None
        train_set = ImageFolder(root=os.path.join(args.datadir, 'imagenet-r'),transform=train_transform)
        test_set = ImageFolder(root=os.path.join(args.datadir, 'imagenet-r'),transform=test_transform)
        class_cnt = 1000
    elif args.dataset == 'imagenet-sketch':
        train_set = ImageFolder(root=os.path.join(args.datadir, 'sketch'),transform=train_transform)
        test_set = ImageFolder(root=os.path.join(args.datadir, 'sketch'),transform=test_transform)
        class_cnt = 1000
    elif args.dataset == 'imagenet-a':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        train_set = ImageFolder(root=os.path.join(args.datadir, 'imagenet-a'),transform=train_transform)
        test_set = ImageFolder(root=os.path.join(args.datadir, 'imagenet-a'),transform=test_transform)
        class_cnt = 1000
    elif args.dataset == 'imagenet-v2':
        from imagenetv2_pytorch import ImageNetV2Dataset
        train_set = ImageNetV2Dataset("matched-frequency")
        test_set = ImageNetV2Dataset("matched-frequency")
        class_cnt = 1000
    else:
        raise NotImplementedError(f"{dataset} not supported")
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    if not args.shuffle:
        train_sampler = SequentialSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set, shuffle=False)
    else:
        train_sampler = RandomSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set)
    test_sampler = SequentialSampler(test_set) if args.local_rank == -1 else DistributedSampler(test_set, shuffle=False)  
    
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True) if test_set is not None else None

    args.class_cnt = class_cnt
    args.normalize = normalize
    args.total_train_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps
    logger.info(f'Dataset information: {dataset}\n')
    logger.info(f'{len(train_set)} images for training \t')
    logger.info(f'{len(test_set)} images for testing\t')

    return train_loader, test_loader


def get_batch(data_loader, batch_index):
    start_index = batch_index * data_loader.batch_size
    end_index = start_index + data_loader.batch_size
    batch_data = []
    batch_targets = []
    
    for i in range(start_index, end_index):
        if i >= len(data_loader.dataset):
            break
        data, target = data_loader.dataset[i]
        batch_data.append(data)
        batch_targets.append(target)
    
    return torch.stack(batch_data), torch.tensor(batch_targets)


def get_indices(full_data):
    full_len = len(full_data)
    train_len = int(full_len * 0.9)
    indices = np.random.permutation(full_len)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    return train_indices, val_indices


class SubsetWithTransform(Subset):
    def __init__(self, dataset, indices, transform=None):
        super(SubsetWithTransform, self).__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y


class SUN397Dataset(Dataset):
    def __init__(self, img_dir, partition_file, target_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(target_file, 'r') as f:
            self.label_names = [l.strip() for l in f.readlines()]
        self.label_idx = {name: idx for idx,name in enumerate(self.label_names)}

        self.img_names = []
        self.targets = []
        with open(partition_file, 'r') as f:
            lines = f.readlines()
        for l in lines:
            l = l.strip()
            self.img_names.append(l)
            label_name, _ = os.path.split(l)
            self.targets.append(self.label_idx[label_name])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.img_dir+img_name
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = self.targets[idx]
        return image, target


class TinyImageNet(Dataset):
    def __init__(self, root_dir, annotations_file, label_ids_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.entries = open(annotations_file).read().strip().split('\n')

        with open(label_ids_file, 'r') as f:
            self.label_names = [l.strip() for l in f.readlines()]
        self.label_names = sorted(self.label_names)
        self.label_idx = {name: idx for idx,name in enumerate(self.label_names)}

        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        line = self.entries[index].split('\t')
        img_path, annotation = line[0], line[1]
        image = Image.open(self.root_dir + '/' + img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        return image, int(self.label_idx[annotation])


class LMDBDataset(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__()
        db_path = os.path.join(root, f"{split}.lmdb")
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = pickle.loads(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf)

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class COOPLMDBDataset(LMDBDataset):
    def __init__(self, root, split="train", transform=None) -> None:
        super().__init__(root, split, transform=transform)
        with open(os.path.join(root, "split.json")) as f:
            split_file = json.load(f)
        idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split_file["test"]}.items()))
        self.classes = list(idx_to_class.values())


class ReverseImageFolder(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        return super().__getitem__(len(self) - 1 - index)  # This reverses the order of items