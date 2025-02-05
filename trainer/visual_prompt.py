import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)

class LoR_VP(nn.Module):
    def __init__(self, args, normalize=None):
        logger.info('prompt method: barfull\n')
        super(LoR_VP, self).__init__()
        width = args.bar_width
        height = args.bar_height
        self.output_size = args.output_size
        self.normalize=normalize
        self.network = args.network

        init_methods = args.init_method.split(',')
        self.left_bar = torch.nn.Parameter(torch.empty(3, height, width))
        self.get_init(init_methods[0], self.left_bar)
        self.right_bar = torch.nn.Parameter(torch.empty(3, width, height))
        self.get_init(init_methods[1], self.right_bar)
        self.program = torch.bmm(self.left_bar, self.right_bar)

        logger.info(f'width: {args.bar_width}, height: {args.bar_height}, output size: {args.output_size}, input size: {args.input_size}, patch size: {args.patch_size}, patch num: {self.patch_num}, mask l pad: {self.mask_l_pad}, mask r pad: {self.mask_r_pad}')

    def get_init(self, init_method, params):
        if init_method == 'zero':
            params.data.fill_(0)
        elif init_method == 'random':
            params.data.normal_(0, 1)
        elif init_method == 'xavier':
            torch.nn.init.xavier_uniform_(params)
        elif init_method == 'kaiming':
            torch.nn.init.kaiming_uniform_(params, nonlinearity='relu')
        elif init_method == 'uniform':
            torch.nn.init.uniform_(params, a=-0.1, b=0.1)
        elif init_method == 'normal':
            torch.nn.init.normal_(params, mean=0.0, std=0.01)

    def forward(self, x):
        self.program = torch.bmm(self.left_bar, self.right_bar)
        x = x + self.program
        if self.normalize is not None:
            x = self.normalize(x)
        return x


