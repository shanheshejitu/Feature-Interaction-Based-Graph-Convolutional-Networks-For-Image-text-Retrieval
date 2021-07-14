import torch
import torch.utils.data as data

from process_data.utils import img_feat_path_load,proc_img_feat,proc_ques,create_glove_dict

import os
import glob
import numpy as np
import codecs



class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, arg, data_split):
        loc = arg.dataset_path + '/'
        #loc = arg.feature_path+ '/'
        self.data_split = data_split
        self.arg = arg

        # Captions
        self.captions = []
        self.ids = []
        with codecs.open(loc + '%s_caps.txt' % data_split, 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。

        with codecs.open(loc + '%s_ids.txt' % data_split, 'r', encoding='utf-8') as f:
            for line in f:
                self.ids.append(line.strip())

        self.images = np.load(loc + '%s_ims.npy' % data_split)

        # self.img_feat_path_list = []
        # Image features
        # if data_split in ['train', 'dev', 'test']:
        #     self.img_feat_path_list += glob.glob(arg.img_feat_path[data_split] + '*.npz')  # !!!
        # self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        self.token_to_ix = create_glove_dict()

        self.length = len(self.captions)

        print("split: %s, total images: %d, total captions: %d" % (data_split, len(self.images), self.length))
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        # img_id = self.ids[int(index / self.im_div)]
        # image = np.load(self.iid_to_img_feat_path[int(img_id)])
        # image = image['x'].transpose((1, 0))
        # image = proc_img_feat(image, 100)     # [100, 2048]
        img_id = int(index / self.im_div)
        image = self.images[img_id]

        caption = self.captions[index]
        caption, le = proc_ques(caption, self.token_to_ix, self.arg.max_length)

        return torch.tensor(image), torch.tensor(caption), torch.tensor(index), torch.tensor(le)

    def __len__(self):
        return self.length


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return  torch.utils.data.dataloader.default_collate(batch)



def get_precomp_loader(arg, data_split, shuffle=True,
                       num_workers=2,batch_size=100):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(arg, data_split)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=arg.batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              )
    return data_loader


def get_loaders(arg):    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    train_loader = get_precomp_loader(arg, 'train',True, arg.num_workers)
    val_loader = get_precomp_loader(arg, 'test',False, arg.num_workers)
    return train_loader, val_loader


def get_test_loader(arg, split_name):

    test_loader = get_precomp_loader(arg, split_name,
                                     False, arg.num_workers)
    return test_loader