import torch
from torch.nn.functional import one_hot
import time
import logging
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def label_mapping_base(logits, mapping_sequence):
    modified_logits = logits[:, mapping_sequence]

    return modified_logits


def get_dist_matrix(fx, y):
    fx = one_hot(torch.argmax(fx, dim = -1), num_classes=fx.size(-1))
    dist_matrix = [fx[y==i].sum(0).unsqueeze(1) for i in range(len(y.unique()))]
    dist_matrix = torch.cat(dist_matrix, dim=1)

    return dist_matrix


def predictive_distribution_based_multi_label_mapping(dist_matrix, mlm_num: int):
    assert mlm_num * dist_matrix.size(1) <= dist_matrix.size(0), "source label number not enough for mapping"
    mapping_matrix = torch.zeros_like(dist_matrix, dtype=int)
    dist_matrix_flat = dist_matrix.flatten() # same memory
    for _ in range(mlm_num * dist_matrix.size(1)):
        loc = dist_matrix_flat.argmax().item()
        loc = [loc // dist_matrix.size(1), loc % dist_matrix.size(1)]
        mapping_matrix[loc[0], loc[1]] = 1
        dist_matrix[loc[0]] = -1
        if mapping_matrix[:, loc[1]].sum() == mlm_num:
            dist_matrix[:, loc[1]] = -1

    return mapping_matrix


def generate_label_mapping_by_frequency(network, data_loader, mapping_num = 1):
    device = next(network.parameters()).device
    if hasattr(network, "label_mapping"):
        network.label_mapping = None
    if hasattr(network, "eval"):
        network.eval()
    fx0s = []
    ys = []
    start = time.time()
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            fx0 = network(x)
        fx0s.append(fx0)
        ys.append(y)
    end = time.time()
    logger.info(f'Label Mapping\tTime {end-start:.2f}')
    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    if ys.size(0) != fx0s.size(0):
        assert fx0s.size(0) % ys.size(0) == 0
        ys = ys.repeat(int(fx0s.size(0) / ys.size(0)))
    dist_matrix = get_dist_matrix(fx0s, ys)
    pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num))
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]

    return mapping_sequence


def generate_label_mapping_by_frequency_ordinary(network, data_loader, mapping_num = 1):
    if hasattr(network, "eval"):
        network.eval()
    fx0s = []
    ys = []
    start = time.time()
    for i, (x, y) in enumerate(data_loader):
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            fx0 = network((x))
        fx0s.append(fx0)
        ys.append(y)
    end = time.time()
    logger.info(f'Label Mapping\tTime {end-start:.2f}')
    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    if ys.size(0) != fx0s.size(0):
        assert fx0s.size(0) % ys.size(0) == 0
        ys = ys.repeat(int(fx0s.size(0) / ys.size(0)))
    dist_matrix = get_dist_matrix(fx0s, ys)
    pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num))
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]

    return mapping_sequence


# Assuming the teacher network is a class that inherits from torch.nn.Module
class CustomNetwork(torch.nn.Module):
    def __init__(self, network, visual_prompt, label_mapping, mapping_sequence, args):
        super(CustomNetwork, self).__init__()
        self.network = network
        self.network.eval()
        frozen_params = []
        for param in self.network.parameters():
            param.requires_grad = False
            frozen_params.append(param)
        frozen_params_num = sum(p.numel() for p in frozen_params)
        logger.info(f"frozen_params_num: {frozen_params_num}")
        self.label_mapping = label_mapping
        self.mapping_sequence = mapping_sequence
        self.downstream_mapping = args.downstream_mapping
        self.normalize = args.normalize
        logger.info(f"normalize: {self.normalize}")
        if args.network in ['vit_b_16', 'vit_l_16']:
            if self.downstream_mapping == 'fm':
                self.downstream_mapping = 'lp'
        
        self.visual_prompt = visual_prompt
        if self.visual_prompt is not None:
            vp_params = []
            for param in self.visual_prompt.parameters():
                param.requires_grad = True
                vp_params.append(param)
            vp_params_num = sum(p.numel() for p in vp_params)
            logger.info(f"vp_params_num: {vp_params_num}")

        if self.downstream_mapping == 'lp':
            lp_params = []
            if 'swin' in args.network:
                self.network.head.fc = torch.nn.Linear(network.head.fc.in_features, args.class_cnt).cuda()
                for param in self.network.head.fc.parameters():
                    param.requires_grad = True
                    lp_params.append(param)
            elif hasattr(network, 'fc'):
                self.network.fc = torch.nn.Linear(network.fc.in_features, args.class_cnt).cuda()
                for param in self.network.fc.parameters():
                    param.requires_grad = True
                    lp_params.append(param)
            elif hasattr(network, 'head'):
                self.network.head = torch.nn.Linear(network.head.in_features, args.class_cnt).cuda()
                for param in self.network.head.parameters():
                    param.requires_grad = True
                    lp_params.append(param)
            elif hasattr(network, 'classifier'):
                self.network.classifier[1] = torch.nn.Linear(network.classifier[1].in_features, args.class_cnt).cuda()
                for param in self.network.classifier[1].parameters():
                    param.requires_grad = True
                    lp_params.append(param)

            lp_params_num = sum(p.numel() for p in lp_params)
            logger.info(f"lp_params_num: {lp_params_num}")

        self.fm = None
        if self.downstream_mapping == 'fm':
            fm_params = []
            if 'swin' in args.network:
                self.out_features = network.head.fc.out_features
            elif hasattr(network, 'fc'):
                self.out_features = network.fc.out_features
            elif hasattr(network, 'head'):
                self.out_features = network.head.out_features
            elif hasattr(network, 'classifier'):
                self.out_features = network.classifier[1].out_features
            self.fm = torch.nn.Linear(self.out_features, args.class_cnt).cuda()
            for param in self.fm.parameters():
                param.requires_grad = True
                fm_params.append(param)
            fm_params_num = sum(p.numel() for p in fm_params)
            logger.info(f"fm_params_num: {fm_params_num}")
        

    def get_tunable_params(self):
        return [param for param in self.parameters() if param.requires_grad]
    

    def forward(self, x):
        if self.visual_prompt is not None:
            x = self.visual_prompt(x)
        elif self.normalize is not None:
            x = self.normalize(x)
        x = self.network(x)
        if isinstance(x, tuple):
            x = x[0]
        if self.downstream_mapping == 'fm' and self.fm is not None:
            x = self.fm(x)
        x = F.log_softmax(x, dim=-1)
        if self.downstream_mapping in ('ilm', 'flm', 'origin') and self.label_mapping is not None:
            x = self.label_mapping(x)
        
        return x
