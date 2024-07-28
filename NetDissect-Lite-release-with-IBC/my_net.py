from torch import nn
from torchvision.models import alexnet
import torch
import torch.nn.functional as F
import torch.nn.init as init


class AlexEncoderDecouple(nn.Module):
    def __init__(self, fea_dim, class_num):
        super(AlexEncoderDecouple, self).__init__()

        if class_num == 397:
            self.alexnet = alexnet(num_classes=365)
            alexnet_cls = alexnet(num_classes=365)
        else:
            self.alexnet = alexnet(pretrained=True)
            alexnet_cls = alexnet(pretrained=True)

        self.class_num = class_num
        output_ch = 256  # feature maps num of output of conv_basic
        self.code_len = fea_dim
        assert self.code_len >= 0, "hash length illegal"

        self.conv_basic = self.alexnet.features
        # the last conv layer for alignment and disentanglement
        self.conv_last = nn.Conv2d(output_ch, self.code_len, kernel_size=3, padding=1)
        init.kaiming_normal(self.conv_last.weight)
        self.conv_last.bias.data.fill_(0.0)

        # the classifier transformation layers of upper real-valued branch
        if self.code_len != output_ch:
            self.fc1 = nn.Linear(self.code_len * 6 * 6, 4096)
            self.fc2 = nn.Sequential(*list(alexnet_cls.classifier.children())[2:6])
            self.fc1.weight.data.normal_(0, 0.01)
            self.fc1.bias.data.zero_()
            self.classifier = nn.Sequential(nn.Dropout(), self.fc1, self.fc2)
        else:
            self.classifier = nn.Sequential(*list(alexnet_cls.classifier.children())[:6])

        self.pre_hash_layer = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        # self.model = nn.Sequential(self.conv_basic, self.pre_hash_layer)

        # hash layer
        self.hash_layer = nn.Linear(4096, self.code_len)

        # classifier for upper real-valued and bottom hash branch
        self.real_prediction = LinearBlock(4096, self.class_num, norm='none', activation='none')
        self.hash_prediction = LinearBlock(self.code_len, self.class_num, norm='none', activation='none')

        self.conv_out = None  # to be aligned real-valued filter output
        self.hash_codes = None  # to be aligned binary codes (hash layer output)
        self.real_logits = None  # category prediction output of real-valued branch
        # self.codes_logits = None  # category prediction output of hash branch

    def forward(self, x):
        x = self.conv_basic(x)
        return F.relu(self.conv_last(x))


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu', use_bias=True):
        super(LinearBlock, self).__init__()
        # use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softmax':
            self.activation = nn.Softmax()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x.view(x.size(0), -1))
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


