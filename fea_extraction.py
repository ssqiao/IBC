from __future__ import print_function
from utils import get_config
from trainer import Trainer
import torch
import torch.backends.cudnn as cudnn
import argparse
import os
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import utils as utl
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/imagenet100.yaml', help='Path to the config file.')
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--save_name', type=str, required=True, help="feature file save name")

opts = parser.parse_args()
torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
# Setup model and data loader
trainer = Trainer(config)

_, test_loader = utl.get_all_data_loaders(config)
model_name = config['model_name']
output_directory = os.path.join("./results", model_name)

if not os.path.exists(output_directory):
    print("Creating directory: {}".format(output_directory))
    os.makedirs(output_directory)

state_dict = torch.load(opts.checkpoint)
trainer.model.load_state_dict(state_dict['state_dict'])
trainer.cuda()
trainer.eval()
encode = trainer.model.encode
GRM = trainer.model.GRM.data.cpu().numpy()

# Start testing
count = 0
sample_num = test_loader.dataset.__len__()
max_num = 50000
fea = []
lab = []
code_len = trainer.model.fea_dim

with torch.no_grad():
    for _, (images, labels) in enumerate(test_loader):
        images = Variable(images.cuda())
        hash_codes = encode(images, masked=False)

        batch_size = images.size(0)

        fea.append(hash_codes)
        lab.append(labels)

        # modify or condition when quantitative test
        count = count + batch_size
        if count >= sample_num or count >= max_num:
            break

    fea = torch.cat(fea).cpu().numpy()
    lab = torch.cat(lab).numpy()

    file_path = os.path.join(output_directory, opts.save_name)

    np.savez(file_path+'.npz', binary_codes=fea, lab=lab, code_length=code_len, GRM=GRM)
    sio.savemat(file_path+'.mat', {'binary_codes': fea, 'lab': lab, 'code_length': code_len, 'GRM': GRM})

