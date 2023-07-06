#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Author   ：Zane
# @Mail     : zaneii@foxmail.com
# @Date     ：2023/6/21 19:11 
# @File     ：utils.py
# @Description :
import os
import pickle
import json
from json import dumps as dump_json
import pandas as pd


def file_encode_ansi_2_uft8(file):
    with open(file, 'r', encoding='ansi') as f:
        content = f.read()

    content_utf8 = content.encode('uft-8')

    with open(file, 'w', encoding='uft-8') as f:
        f.write(content_utf8.decode('utf-8'))


def read_json_file(file_location):
    with open(file_location) as f:
        return json.load(f)


def write_json_file(path, data, w_r_mode='w'):
    with open(path, w_r_mode) as f:
        f.write(dump_json(data))


def json_2_excel(path, is_save=False):
    file_name = lambda x: (x[-5:] == '.json')
    df = pd.read_json(path)
    if is_save:
        df.to_excel(f'{file_name}.xlsx')
    return df


def read_dir_all_file(path=None, all_files_path=[]):
    '''
    从根目录读取所有文件
    :param all_files_path:
    :param path:
    :return:
    '''
    for root, dirs, files in os.walk(path, topdown=True):
        # print('root:',root)
        # print('dirs:',dirs)
        # print('files:',files)
        # print('\n')
        if len(files) > 0:
            each_folder_files = [os.path.join(root, x) for x in files]
            all_files_path.extend(each_folder_files)

        for dir in dirs:
            read_dir_all_file(os.path.join(root, dir),all_files_path)
        return all_files_path


def serialization(data, path):
    """
    :param data: data
    :param path: data.pickle
    :return:
    """
    # 将对象序列化并写入文件
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def unserialization(path):
    """
    :param path: data.pickle
    :return: data
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data