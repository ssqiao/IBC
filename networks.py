from torch import nn
from torchvision.models import alexnet
import torch
import torch.nn.functional as F
import torch.nn.init as init


##################################################################################
# Generator, qss add hierarchy nested gaussian distributions for sampling styles
##################################################################################
class IBC(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, params):
        super(IBC, self).__init__()
        self.fea_dim = params['code_len']
        self.class_num = params['class_num']

        is_sun = params['is_sun']
        weights_file = None
        if 'weights_file' in params.keys():
            weights_file = params['weights_file']
        self.enc_style = AlexEncoderDecouple(self.fea_dim, self.class_num, is_sun, weights_file)

        self.multi_label = False
        if 'multi_label' in params.keys():
            self.multi_label = params['multi_label']

        self.class_weights = None
        if 'class_weights' in params.keys():
            self.class_weights = params['class_weights']

        # Gated Rule Matrix
        self.GRM = nn.Parameter(torch.rand(self.class_num, self.fea_dim))
        self.thr_g = params['thr_g']

    def forward(self, images):
        hash_codes = self.encode(images)
        return hash_codes

    def encode(self, images, masked=True, train_labels=None):
        # encode an image to its binary codes
        codes = self.enc_style(images, masked=masked, GRM=self.GRM, train_labels=train_labels)
        return codes


##################################################################################
# CNN backbone
##################################################################################
# weights_file = '/home/ouc/data1/qiaoshishi/datasets/SUN_attributes/data_256/pretrained_places_models/' \
#                          'pytorch_model/alexnet_places365.pth.tar'
class AlexEncoderDecouple(nn.Module):
    def __init__(self, fea_dim, class_num, is_sun=False, weights_file=None):
        super(AlexEncoderDecouple, self).__init__()
        # TODO init from places365 model
        if not is_sun:
            self.alexnet = alexnet(pretrained=True)
            alexnet_cls = alexnet(pretrained=True)
        else:
            self.alexnet = alexnet(num_classes=365)
            alexnet_cls = alexnet(num_classes=365)
            model_file = weights_file
            checkpoint = torch.load(model_file)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            self.alexnet.load_state_dict(state_dict)
            alexnet_cls.load_state_dict(state_dict)
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

    def forward(self, x, masked=True, GRM=None, train_labels=None):
        self.real_logits = None
        x1 = self.conv_basic(x)

        x = x1.view(x1.size(0), -1)
        x = self.pre_hash_layer(x)
        real_code = self.hash_layer(x)
        codes = F.sigmoid(real_code)

        self.conv_out = F.relu(self.conv_last(x1))
        self.hash_codes = codes.unsqueeze(2).unsqueeze(3)

        # qss TODO control subsequent classification run or not
        # return codes

        if masked:
            out = self.conv_out.clone()

            tmp_lab = train_labels[:, 0].long()
            legal_flag = tmp_lab.ne(99999)
            if not legal_flag.sum():  # all class labels are invalid
                # TODO
                out = out.view(out.size(0), -1)
                out = self.classifier(out)
                self.real_logits = self.real_prediction(out)
                return codes
            legal_index = legal_flag.nonzero().squeeze()
            mask = GRM[tmp_lab[legal_index]].unsqueeze(2).unsqueeze(3)
            out[legal_index] = mask * out[legal_index]

            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            self.real_logits = self.real_prediction(out)
            return codes

        out = self.conv_out.view(self.conv_out.size(0), -1)
        out = self.classifier(out)
        self.real_logits = self.real_prediction(out)
        return codes

    def get_real_logits(self):
        return self.real_logits


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

