from __future__ import print_function
import os
import pickle


import logging
import numpy
from process_data.data import get_test_loader
import time
import numpy as np
import torch
from networkgraph1 import Net
from collections import OrderedDict


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()): # drop iter
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items(): # drop iter
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(arg, model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """

    # switch to evaluate mode
    model.eval()

    start = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_masks = None
    print("start loading val data...")
    device = torch.device("cuda:{}".format(arg.device))
    for i, (img, txt, index, le) in enumerate(data_loader):

        img = img.to(device)
        txt = txt.to(device)
        le = le.to(device)
        # index = index.to(device)


        # compute the embeddings
        img_emb, cap_emb, img_feat_mask, txt_feat_mask = model.forward_emb(img, txt, le)
        # logging("forward finish!")
        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:

            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))    # !!!!!!!!
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1), cap_emb.size(2)))       # 嵌入向量
            img_masks = np.zeros((len(data_loader.dataset), 1, 1, img_feat_mask.size(3))) # , dtype=int
            txt_masks = np.zeros((len(data_loader.dataset), 1, 1, txt_feat_mask.size(3)))


        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[index] = img_emb.data.cpu().numpy().copy()
        cap_embs[index] = cap_emb.data.cpu().numpy().copy()
        img_masks[index] = img_feat_mask.data.cpu().numpy().copy()
        txt_masks[index] = txt_feat_mask.data.cpu().numpy().copy()

        # l_list = [int(l_now) for l_now in le]
        # cur_mask = np.zeros((le.size(0), img_emb.size(1)), dtype=int)
        # for mask_idx, mask_l in enumerate(l_list):
        #     cur_mask[mask_idx, :mask_l] = 1
        #
        # cap_masks[index] = cur_mask    # 在我们的程序中应该用不着
        # measure accuracy and record loss
        # model.forward_loss(img_emb, cap_emb)
        # measure elapsed time

        del img, txt
    end = time.time()
    print('encode_data finished in {}s'.format(int(end - start)))

    return img_embs, cap_embs, img_masks, txt_masks#, cap_masks


def score(arg, model, img_embs, cap_embs, img_masks, txt_masks, shard_size=50):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    print("start compute score...")
    start = time.time()
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])    # !!!!!!!!!!!!!!!!!!!!!!!
    img_masks = np.array([img_masks[i] for i in range(0, len(img_masks), 5)])  # !!!!!!!!!!!!!!!!!!!!!!!

    n_img_shard = int((len(img_embs) - 1) / shard_size) + 1
    n_cap_shard = int((len(cap_embs) - 1) / shard_size) + 1

    score = np.zeros((len(img_embs), len(cap_embs)))

    device = torch.device("cuda:{}".format(arg.device))


    for i in range(n_img_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            im_emb = torch.from_numpy(img_embs[im_start:im_end]).to(device)
            im_mas = torch.from_numpy(img_masks[im_start:im_end]).to(device)
            cap_emb = torch.from_numpy(cap_embs[cap_start:cap_end]).to(device)
            cap_mas = torch.from_numpy(txt_masks[cap_start:cap_end]).to(device)
            im_emb = im_emb.float()
            cap_emb = cap_emb.float()
            im_mas = im_mas.byte()
            cap_mas = cap_mas.byte()

            sim = model.sim2(im_emb, cap_emb, im_mas, cap_mas)  # 然后计算相似性    #[50,50]

            score[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()  #[1000,5000]  逐步填充满score矩阵

        #print("\rid of img_embs "+str(i), end='          ')
    end = time.time()
    print('compute score finished in {}s'.format(int(end - start)))
    print(score.shape)

    #g = score[[0, 1, 2,3,4]]
    #g=g[:,[0,1,2,50,51,52]]
    #print("score2")
    #print(g)
    return score


def evalrank(arg, model, data_loader, state_dict=None, model_path=None, fold5=False, return_ranks=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    #print('Loading model')
    # load model state
    #if state_dict is None:
    #   checkpoint = torch.load(model_path)
    #    model.load_state_dict(checkpoint['state_dict'])
    #else:
    #    model.load_state_dict(state_dict)

    print('Computing results...')
    with torch.no_grad():
        img_embs, cap_embs, img_mas, cap_mas = encode_data(arg, model, data_loader)
        print('Images: %d, Captions: %d' %
              (img_embs.shape[0] / 5, cap_embs.shape[0]))

        if not fold5:
            # no cross-validation, full evaluation
            sc = score(arg, model, img_embs, cap_embs, img_mas, cap_mas)

            #print(list(sc.size()))
            #print(sc.shape)
            #g=np.argsort(-sc,axis=1)
            #g=g[[0,1,2,3,4]]
            #g=g[:,[0,1,2,5,6,7,10,11,12]]
            #g=g[:,[0,1,2,3,4]]
            #g=g[[1,  3, 4,5,6]]
            #print(g)


            r, rt = i2t(img_embs, cap_embs, 1, sc, return_ranks=True)
            ri, rti = t2i(img_embs, cap_embs, 1, sc, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f" % rsum)
            print("Average i2t Recall: %.1f" % ar)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
            print("Average t2i Recall: %.1f" % ari)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
            logfile = open(
                arg.log_path +
                'log_run_' + str(arg.log_id) + '.txt',
                'a+'
            )
            logfile.write(
                'rsum = ' + str(rsum) +
                ' Image to text = ' + str(r) +
                ' text to image = ' + str(ri) +
                '\n\n'
            )
            logfile.close()
            #message = "split: %s, Image to text: (%.1f, %.1f, %.1f) " % (split, r[0], r[1], r[2])
            #message += "Text to image: (%.1f, %.1f, %.1f) " % (ri[0], ri[1], ri[2])
            #message += "rsum: %.1f\n" % rsum
        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            for i in range(5):

                img_emb_i = img_embs[i * 5000:(i + 1) * 5000]

                cap_emb_i = cap_embs[i * 5000:(i + 1) * 5000]

                sc = score(arg, model, img_emb_i, cap_emb_i)
                r, rt0 = i2t(img_emb_i, cap_emb_i, 1, sc, return_ranks=True)
                print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
                ri, rti0 = t2i(img_emb_i, cap_emb_i, 1, sc, return_ranks=True)
                print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

                if i == 0:
                    rt, rti = rt0, rti0
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]

            print("-----------------------------------")
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            print("rsum: %.1f" % (mean_metrics[10] * 6))
            print("Average i2t Recall: %.1f" % mean_metrics[11])
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[:5])
            print("Average t2i Recall: %.1f" % mean_metrics[12])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[5:10])
            #message = "split: %s, Image to text: (%.1f, %.1f, %.1f) " % (split, mean_metrics[0], mean_metrics[1], mean_metrics[2])
            #message += "Text to image: (%.1f, %.1f, %.1f) " % (mean_metrics[5], mean_metrics[6], mean_metrics[7])
            #message += "rsum: %.1f\n" % (mean_metrics[10] * 6)
    if return_ranks:
        return rt, rti


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)