import numpy as np
np.set_printoptions(threshold=100000)
#import pandas as pd
import cv2

# loc = '/home/cz/czf/retrieval/reasoning/datasets/coco_precomp' + '/'
# data_split = 'test'
# images = np.load(loc + '%s_ims.npy' % data_split)
# print(len(images))
#
# img_id = 1
# image = images[img_id]
# print(image[:,0:2])




import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import json

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = "/home/gaofl/datasets/test2014_36/trainval_resnet101_faster_rcnn_genome_36.tsv"

if __name__ == '__main__':

    # Verify we can read a tsv
    in_data = {}
    with open(infile, "r+t") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            #for field in ['boxes', 'features']:
            for field in ['boxes']:
                item[field] = np.frombuffer(base64.decodestring(bytes(item[field],'utf-8')),
                                            dtype=np.float32).reshape((item['num_boxes'], -1)).tolist()
            in_data[item['image_id']] = item

            #break
    #keys = list(in_data.keys())
    #print(keys)
    print(in_data[483108])
    #print(in_data)





    #print(in_data[518742]['image_w'])
    #with open("/home/gaofl/datasets/2.json", "w") as f:
        #f.write(json.dumps(in_data[518742], ensure_ascii=False, indent=4, separators=(',', ':')))

    #d = np.load("/home/cz/czf/VQA/datasets/coco_extract/val2014/COCO_val2014_000000235836.jpg.npz")
    #print(list(d.keys())) # ['x', 'image_w', 'bbox', 'num_bbox', 'image_h']
    #print((d['x'].T[:, 0:2]))
    #print(d['bbox'])

# s = np.load("/home/cz/czf/retrieval/reasoning/save3/score.npy")
# print(s.T.shape)
# s1 = s.T[500]    # 6，46 检索文本
# print(np.argmax(s1))
# ind = np.argpartition(s1, -5)[-5:]
# print(ind[np.argsort(s1[ind])]) # 降序