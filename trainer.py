"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import IBC
from utils import weights_init, get_model_list, get_scheduler, multilabel_to_random_singlelabel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os


class Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.model = IBC(hyperparameters['model'])

        # Set up the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        params_list = [{'params': self.model.enc_style.conv_basic.parameters()},
                       {'params': self.model.enc_style.pre_hash_layer.parameters()},
                       {'params': self.model.enc_style.hash_layer.parameters(), 'lr': lr * 10.},
                       {'params': self.model.enc_style.hash_prediction.parameters(), 'lr': lr * 10.}]
        params_list = params_list+[{'params': self.model.enc_style.conv_last.parameters(), 'lr': lr*10.}] + \
                      [{'params': self.model.enc_style.classifier.parameters()}] + \
                      [{'params': self.model.enc_style.real_prediction.parameters(), 'lr': lr*10.}]

        params_list = params_list+[{'params': self.model.GRM, 'lr': lr*10.}]

        # self.opt = torch.optim.SGD(params_list, lr=lr, momentum=0.9, weight_decay=5e-4)
        self.opt = torch.optim.Adam(params_list, lr=lr, betas=(beta1, beta2),
                                    weight_decay=hyperparameters['weight_decay'])
        self.scheduler = get_scheduler(self.opt, hyperparameters)

        for mod_ in self.model.children():
            name = mod_.__class__.__name__
            if name == 'AlexEncoderDecouple':
                mod_.hash_layer.apply(weights_init(hyperparameters['init']))
                mod_.hash_prediction.apply(weights_init(hyperparameters['init']))
                mod_.real_prediction.apply(weights_init(hyperparameters['init']))

    def forward(self, x):
        self.eval()
        b = self.model.encode(x, masked=False)
        self.train()
        return b

    # qss
    def model_update(self, x, hyperparameters, labels):
        self.opt.zero_grad()

        x_split = torch.split(x, x.shape[0] // 2)
        labels_split = torch.split(labels, labels.shape[0] // 2)

        b_std = self.model.encode(x_split[0], masked=False)
        logit_real_std = self.model.enc_style.get_real_logits()
        self.loss_real_cls_std = self.compute_real_fea_cls_loss(logit_real_std, labels_split[0],
                                                                multi_label=self.model.multi_label)
        self.loss_code_cls_std = self.compute_cls_loss(b_std, labels_split[0], multi_label=self.model.multi_label)
        if self.model.multi_label:
            half_batch_labels = multilabel_to_random_singlelabel(labels_split[1])
            cls_loss_type = 'BCE'
        else:
            half_batch_labels = labels_split[1]
            cls_loss_type = 'CE'
        b_mask = self.model.encode(x_split[1], masked=True, train_labels=half_batch_labels)
        logit_real_mask = self.model.enc_style.get_real_logits()

        self.loss_real_cls_mask = self.compute_real_fea_cls_loss(logit_real_mask, half_batch_labels, loss=cls_loss_type)
        self.loss_code_cls_mask, self.loss_rule_sparse = self.compute_cls_loss(b_mask, half_batch_labels, masked=True,
                                                                               loss=cls_loss_type)

        self.loss_kl_div = self.compute_kl_div(self.model.enc_style.hash_codes, self.model.enc_style.conv_out)

        self.loss_code_sparse = self.compute_sparseness(b_mask, sparse_ratio=0.2)

        self.loss_total = hyperparameters['hash_std_w'] * self.loss_code_cls_std + \
                          hyperparameters['hash_mask_w'] * self.loss_code_cls_mask + \
                          hyperparameters['GRM_w'] * self.loss_rule_sparse + \
                          hyperparameters['align_w'] * self.loss_kl_div + \
                          hyperparameters['real_std_w'] * self.loss_real_cls_std + \
                          hyperparameters['real_mask_w'] * self.loss_real_cls_mask + \
                          hyperparameters['sparse_w'] * self.loss_code_sparse

        self.loss_total.backward()
        self.opt.step()

        with torch.no_grad():
            max_norm, _ = torch.max(torch.abs(self.model.GRM), 0)
            self.model.GRM.set_(torch.div(self.model.GRM, max_norm))
            self.model.GRM.clamp_(0, 1)

    def compute_sparseness(self, codes, sparse_ratio=0.1):
        return F.relu(torch.mean(torch.norm(codes, 1, 1)) - sparse_ratio * codes.shape[1])

    # qss
    def compute_kl_div(self, codes, conv_out):
        feature = F.avg_pool2d(conv_out, conv_out.shape[3])
        return F.kl_div(F.log_softmax(feature, 1), F.softmax(codes, 1))

    def compute_category_loss(self, logits, labels, weights=None, multi_label=False, loss='CE'):
        if multi_label or loss is 'BCE':
            if labels.ndim == 1:
                labels = torch.eye(self.model.class_num).cuda()[labels]
            return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=weights) if labels.size(0) else 0.  # qss pos_weight
        else:
            return F.cross_entropy(logits, labels, weight=weights) if labels.size(0) else 0.

    def compute_real_fea_cls_loss(self, logits, labels, multi_label=False, loss='CE'):
        loss_cls_c = 0.
        if self.model.class_weights is None:
            w = None
        else:
            w = torch.Tensor(self.model.class_weights).cuda()

        tmp_lab = labels[:, 0].long()
        legal_flag = tmp_lab.ne(99999)
        if not legal_flag.sum():
            return loss_cls_c
        legal_index = legal_flag.nonzero().squeeze()
        if multi_label:
            tmp_lab = labels

        loss_cls_c += self.compute_category_loss(logits[legal_index], tmp_lab[legal_index], weights=w,
                                                 multi_label=multi_label, loss=loss)
        return loss_cls_c

    def compute_cls_loss(self, codes, labels, masked=False, multi_label=False, loss='CE'):
        loss_cls = 0.
        loss_G = 0.

        if masked:
            loss_G += F.relu(torch.sum(torch.abs(self.model.GRM)) - self.model.thr_g)

        if self.model.class_weights is None:
            w = None
        else:
            w = torch.Tensor(self.model.class_weights).cuda()

        tmp_lab = labels[:, 0].long()
        feature = codes
        legal_flag = tmp_lab.ne(99999)
        if not legal_flag.sum():
            if masked:
                return loss_cls, loss_G
            else:
                return loss_cls

        legal_index = legal_flag.nonzero().squeeze()
        if multi_label:
            tmp_lab = labels
        # qss
        if masked:
            mask = self.model.GRM[tmp_lab[legal_index]]
            fea = feature.clone().squeeze()
            fea[legal_index] = torch.mul(fea[legal_index], mask)
            fea_predict = self.model.enc_style.hash_prediction(fea)
        else:
            fea_predict = self.model.enc_style.hash_prediction(feature)

        loss_cls += self.compute_category_loss(fea_predict[legal_index], tmp_lab[legal_index], weights=w,
                                               multi_label=multi_label, loss=loss)

        if masked:
            return loss_cls, loss_G
        else:
            return loss_cls

    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    # qss, gen, dis model
    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "checkpoint")
        state_dict = torch.load(last_model_name)
        self.model.load_state_dict(state_dict['state_dict'])
        iterations = int(last_model_name[-11:-3])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.opt.load_state_dict(state_dict['optimizer'])
        # Reinitilize schedulers
        self.scheduler = get_scheduler(self.opt, hyperparameters, iterations)

        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save model and optimizers
        model_name = os.path.join(snapshot_dir, 'checkpoint_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'state_dict': self.model.state_dict()}, model_name)
        torch.save({'optimizer': self.opt.state_dict()}, opt_name)

