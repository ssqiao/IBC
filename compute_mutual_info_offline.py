import time
import argparse
import struct
import numpy as np
import scipy.io as sio
import multiprocessing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.feature_selection import mutual_info_classif
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num', type=int, default=100, help='Class num.')
    parser.add_argument('--fea_dim', type=int, default=256, help='Dim of feature.')
    parser.add_argument('--sample_num', type=int, default=5000, help='Sample num.')
    parser.add_argument('--fea_file_path', type=str, default='.', help="feature file path")
    parser.add_argument('--label_file_path', type=str, default='.', help="class label file path")
    parser.add_argument("--is_mat_file", action="store_true")
    parser.add_argument("--is_sun", action="store_true")
    opts = parser.parse_args()
    feature = np.zeros((opts.sample_num, opts.fea_dim), dtype=np.float32)
    label = np.zeros((opts.sample_num,))
    if opts.is_mat_file:
        tmp_fea = sio.loadmat(opts.fea_file_path)
        feature = tmp_fea['binary_codes']  # TODO, change the name based on methods
        feature = np.sign(feature-0.5).squeeze()
        tmp_lab = sio.loadmat(opts.label_file_path)
        label = tmp_lab['lab'][:, 0].squeeze()  # TODO, change the name based on methods
        # tmp_lab = h5py.File(opts.label_file_path)
        # label = tmp_lab['test_lab'][:].transpose().squeeze()
    else:
        f_fea = open(opts.fea_file_path, 'rb')
        f_lab = open(opts.label_file_path, 'rb')
        for i in range(opts.sample_num):
            label[i] = struct.unpack('f', f_lab.read(4))[0]
            for j in range(opts.fea_dim):
                feature[i, j] = struct.unpack('f', f_fea.read(4))[0]

        f_fea.close()
        f_lab.close()

    t_start = time.time()
    p_thr = multiprocessing.Pool(8)
    mutual_info_matrix = []
    res_list = []
    if opts.is_sun:
        max_num = 20  # ensure pos vs neg 1:1, 100 for imagenet, 20 for sun
    else:
        max_num = 100
    for lab in range(opts.class_num):
        print(lab)
        fea_tmp = []
        lab_tmp = np.zeros((max_num,))
        idx_p = np.argwhere(label == lab)
        if not len(idx_p):
            continue
        p_num = min(max_num//2, len(idx_p))
        idx_n = np.argwhere(label != lab)
        n_num = len(idx_n)
        rand_select_n = np.random.permutation(np.arange(n_num))
        fea_tmp.append(feature[idx_p[:p_num]].squeeze(1))
        fea_tmp.append(feature[idx_n[rand_select_n[:max_num - p_num]]].squeeze(1))
        fea_tmp = np.concatenate(fea_tmp)
        lab_tmp[:p_num] = 1
        # lab_tmp[p_num:] = group_lab[idx_n[rand_select_n[:max_num-p_num]], 0].squeeze(1)
        res = p_thr.apply_async(mutual_info_classif, (fea_tmp, lab_tmp,))
        # res = p_thr.apply_async(compute_MI, (group_fea, group_lab[:, 0], lab,))
        res_list.append(res)
    p_thr.close()
    p_thr.join()
    t_end = time.time()
    print(t_end-t_start)
    for res in res_list:
        mutual_info_matrix.append([res.get()])
    mutual_info_matrix = np.concatenate(mutual_info_matrix)
    print(mutual_info_matrix.max(axis=0).mean())
