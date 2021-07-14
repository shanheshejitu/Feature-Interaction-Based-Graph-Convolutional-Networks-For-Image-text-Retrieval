import numpy as np
import re
import os

en_stop_fname = os.path.join(os.path.dirname(__file__), 'stopwords_en.txt')
ENGLISH_STOP_WORDS = set(map(str.strip, open(en_stop_fname).readlines()))
def img_feat_path_load(path_list):  # 地址列表哪来的
    iid_to_path = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        iid_to_path[iid] = path

    return iid_to_path    # 得到一个字典，key为图片 id，value为图片地址


def proc_img_feat(img_feat, img_feat_pad_size):   # 固定特征个数，不够就 pad
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat


def proc_ques(ques, token_to_ix, max_token):  # 查字典，把单词变为标号
    ques_ix = np.zeros(max_token, np.int64)

    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques.lower()
    ).replace('-', ' ').replace('/', ' ').split()

    le = len(words)

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']   # ！！！！！！！！！！！！！！！

        if ix + 1 == max_token:
            break
    #ques_ixs=[x for x in ques_ix]
    ques_ixs=[x for x in ques_ix if x not in ENGLISH_STOP_WORDS]
    return ques_ixs, min(le, max_token)


def create_glove_dict():
    glove_file = "/home/gaofl/datasets/glove.6B.300d.txt"
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    with open(glove_file, encoding="utf-8") as f:
        id = 2
        while 1:
            l = f.readline()
            if l == '':
                break
            vals = l.split(' ')
            word = vals[0]
            token_to_ix[word] = id
            id += 1
    return token_to_ix