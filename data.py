import torch.utils.data as data
import os.path
from PIL import Image
from numpy import random
import torch


# open an image & convert to RGB format
def default_loader(path):
    tmp = Image.open(path)
    return tmp.convert('RGB')


# read txt list path+label
def default_flist_reader1(flist):
    """
    flist format: impath label\n impath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()  # delete space char at start and end locations
            imlist.append(impath)

    return imlist


# filter and leave the desired attributes
def default_flist_reader(data_dir, file_name, filter_lab, shuffle=False):
    file_path = os.path.join(data_dir, file_name)
    print('Getting file list!')
    with open(file_path) as fid:
        content = fid.read()
        contentList = content.split('\n')
    imgs = []
    for term in contentList:
        tmp = term.split(' ')
        img_path = data_dir + tmp[0]
        labs = tmp[1:]
        label = []
        for idx, lab in enumerate(labs):
            if filter_lab is not None and idx in filter_lab:
                continue
            elif lab is not '':
                if int(lab) == -1:
                    label.append(0)
                # 1 or 2 (missing)
                else:
                    label.append(int(lab))
        imgs.append([img_path, label])
    if shuffle:
        random.shuffle(imgs)
    return imgs


# inherit from abstract class Dataset and override the getitem and len func.
# root path, txt path (contain image path & labels), txt list reader, image loader
class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, filter_label, transform=None, shuffle=False,  extra_attr_file=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.filter_label = filter_label
        self.imgs = flist_reader(self.root, flist, self.filter_label, shuffle)
        self.extra_attr_file = extra_attr_file
        # qss attribute label file is independent
        if self.extra_attr_file is not None:
            self.imgs_e = flist_reader(self.root, self.extra_attr_file, self.filter_label, shuffle)
            self.imgs += len(self.imgs)/len(self.imgs_e) * self.imgs_e
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        term = self.imgs[index]
        img = self.loader(term[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.FloatTensor(term[1])

    def __len__(self):
        return len(self.imgs)


