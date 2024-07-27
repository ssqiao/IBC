from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageLabelFilelist
import torch
import os
import math
import yaml
import torch.nn.init as init
import time
# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_config                : load yaml file
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_loss                : write records into tensorboard
# get_model_list            : Get model list for resume
# get_scheduler             : set the learning rate scheduler for optimization
# weights_init              : get init func for network weights


def get_all_data_loaders(conf, test_shuffle=False):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    root = conf['data_folder']
    filter_label = conf['filter_labels']
    if filter_label == []:
        filter_label = None
    train_file = conf['train_list_file']
    test_file = conf['test_list_file']
    if 'new_size' in conf:
        new_size_a = conf['new_size']
        new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    crop = False if height == new_size_a and width == new_size_b else True
    horizon = True
    extra_attr_file = conf['train_attr_list_file'] if 'train_attr_list_file' in conf.keys() else None

    train_data_loader = get_data_loader_list(root, train_file, filter_label, batch_size, True,
                                             new_size_a, extra_attr_file, height, width, num_workers, crop, horizon)
    test_data_loader = get_data_loader_list(root, test_file, filter_label, batch_size, False, new_size_a,
                                            extra_attr_file, height, width, num_workers, crop, horizon, test_shuffle)
    return train_data_loader, test_data_loader


def get_data_loader_list(root, file_list, filter_label, batch_size, train, new_size=None, extra_attr_file=None,
                         height=128, width=128, num_workers=4, crop=True, horizon=True, pre_shuffle=True):
    transform_list = [transforms.ToTensor(),  # [0,255] to [0,1]
                      transforms.Normalize((0.485, 0.456, 0.406),
                                           (0.229, 0.224, 0.225))]  # [0,1] to [-1,1]
    # transform_list = [transforms.Resize(
    #     (new_size, new_size))] + transform_list if new_size is not None else transform_list  # resize offline
    if crop:
        if train:
            transform_list = [transforms.RandomCrop((height, width))] + transform_list
        else:
            transform_list = [transforms.CenterCrop((height, width))] + transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train and horizon else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageLabelFilelist(root, file_list, filter_label, transform=transform, shuffle=pre_shuffle,
                                 extra_attr_file=extra_attr_file)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers)
    return loader


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def multilabel_to_random_singlelabel(raw_labels):
    tar_labels = raw_labels.clone().detach()
    for i, (lab) in enumerate(raw_labels):
        labels = lab.nonzero().squeeze(1)
        tar_labels[i, 0] = labels[torch.randperm(labels.shape[0])[0]]
    return tar_labels


def prepare_sub_folder(output_directory):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) if not callable(getattr(trainer, attr)) and not attr.startswith("__") and
               ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
              os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if models is None:
        return None
    models.sort()
    last_model_name = models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
