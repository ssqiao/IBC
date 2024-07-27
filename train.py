from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, Timer
import argparse
from trainer import Trainer
import torch.backends.cudnn as cudnn
import torch
import os
import sys
import tensorboardX
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/imagenet100.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

seed = 1245
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']

train_loader, test_loader = get_all_data_loaders(config)

# qss, Setup model and data loader
trainer = Trainer(config)

trainer.cuda()

# Setup logger and output folders
model_name = config['model_name']
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images, labels) in enumerate(train_loader):
        # Main training code
        trainer.update_learning_rate()
        images = images.cuda().detach()
        labels = labels.cuda().detach()
        with Timer("Elapsed time in update: %f"):
            trainer.model_update(images, config, labels)
            torch.cuda.synchronize()

        # Dump training stats in log file, ok
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Save network weights, ok
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

