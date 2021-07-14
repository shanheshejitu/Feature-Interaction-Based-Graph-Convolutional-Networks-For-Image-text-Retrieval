# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:20:58 2020
@author: Tengfei Liu
"""

import re
import requests
import urllib.request
import os
import argparse
import io
import sys
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
#mag_text = r'E:\mag\mag\test20000-25000.txt'
mag_text = 'train60000-65000.txt'
#构造基本的爬虫地址
url = 'https://www.sci-hub.ren/'
#第一步，获取MAG每一个样本的标题
num = 0
f2 = open(r'train60000-65000_has_paper.txt','a')
f3 = open(r'train60000-65000_no_paper.txt','a')
with open(mag_text,'r',encoding='utf-8') as f:
    for line in f.readlines(): 
        num = num + 1
        if num >= 0 and num <= 5000:
            print(num)
            text = eval(line)['text'].strip()    #读取出来的文本样本
            params = {'request': text}
            try:
                html = requests.post(url, params)
                pdf_url = re.findall(r"(?<=src = \").+?(?=\" id = \"pdf\")", html.text)
                if len(pdf_url) == 0:
                    f3.writelines(line)
                else:
                    f2.writelines(line)
            except:
                continue
        else:
            continue
f2.close()
f3.close()

# mag_train = r'E:\mag\mag\test1-5000.txt'
# url = 'https://www.sci-hub.ren/'
# #第一步，获取MAG每一个样本的标题
# localDir = 'E:/mag/MAG-Train/'
# if not os.path.exists(localDir):
#     os.makedirs(localDir)
# num = 0
# # data_mat = []
# # with open('no_result_test.txt','r') as f_yan:
# #     for line in f_yan.readlines():
# #         data_mat.append(line.strip())

# with open(mag_train,'r',encoding='utf-8') as f:
#     for line in f.readlines():
#         num = num + 1
#         if num >= 0 and num < 100:     #1588
#             print(num)
#             text = eval(line)['text'].strip()    #读取出来的文本样本
#             label = eval(line)['label'].strip()
#             file_path = os.path.join(localDir, text + '.pdf')
#             label_path = os.path.join(localDir, text + '.txt')
#             if os.path.exists(file_path):
#                 print('File 【{}.pdf】 exists,skip downloading.'.format(text))
#                 continue
#             else:
#                 params = {'request': text}
#                 try:
#                     html = requests.post(url, params)
#                     pdf_url = re.findall(r"(?<=src = \").+?(?=\" id = \"pdf\")", html.text)
#                     if len(pdf_url) == 0:
#                         f_no_result = open('no_result.txt','w')
#                         f_no_result.write(text + '\n')
#                         f.close()
#                         continue
#                     else:
#                         try:
#                             print('['+str(num)+"]  Downloading -> " + file_path)
#                             r = requests.get(pdf_url[0])
#                             fo = open(file_path,'wb')                         # 注意要用'wb',b表示二进制，不要用'w'
#                             fo.write(r.content)                               # r.content -> requests中的二进制响应内容：以字节的方式访问请求响应体，对于非文本请求
#                             fo.close()
#                             with open(label_path, 'w') as f_label:
#                                 f_label.write(label.lower())
#                         except:
#                             continue
#                 except:
#                     continue