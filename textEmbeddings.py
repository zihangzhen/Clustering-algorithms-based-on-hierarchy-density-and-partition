#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Author   ：Zane
# @Mail     : zaneii@foxmail.com
# @Date     ：2023/6/19 16:33 
# @File     ：textEmbeddings.py
# @Description : bert: https://huggingface.co/bert-base-uncased

from transformers import BertTokenizer, BertModel
from itertools import chain

import torch
import os
from utils import file_encode_ansi_2_uft8


class textEmbeddings:
    def __init__(self, model='bert_base'):
        self.model = model
        self.dataset_path = './'
        # self.model = "text2vec_chinese"

        # 单GPU或者CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

    def _get_data_contents(self, truncated_leagth, order=True):
        # 读文件
        dirs = os.listdir(self.dataset_path)
        contents = [] # 按顺序读取的全部文件
        contents_by_dir = []  # 按照目录分类的全部文件
        for dir in dirs:
            dir_path = self.dataset_path + dir
            files = os.listdir(dir_path)
            tmp = [] # 按照目录分类的文件
            for file in files:
                file_p = os.path.join(dir_path, file)
                try:
                    with open(file_p, 'r', encoding='utf-8') as f:
                        a = f.read()[:truncated_leagth]
                        tmp.append(a)
                except UnicodeDecodeError:
                    print(file_p)
            contents_by_dir.append(tmp)

        # 排序
        if order: # 默认按照文件目录的先后顺序排列文件
            for files in contents_by_dir:
                contents.extend(files)
        else: # 按照“你一个我一个”的顺序排列
            contents = list(chain.from_iterable(zip(*contents_by_dir)))
        return contents

    def _embeddings(self, contents, max_length):
        # Mean Pooling - Take attention mask into account for correct averaging
        tokenizer = BertTokenizer.from_pretrained(self.model)
        model = BertModel.from_pretrained(self.model)
        # Tokenize sentences
        encoded_input = tokenizer(contents, padding=True, truncation=True, return_tensors='pt', max_length=max_length)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def embedding_dataset(self, dataset, truncated_leagth=200, max_length=200, order=True):
        """
        对数据集目录下子目录内的所有文件进行词嵌入
        :param dataset:
        :param truncated_leagth:
        :param max_length:
        :return:
        """
        self.dataset_path = dataset
        contents = self._get_data_contents(truncated_leagth=truncated_leagth,order=order)
        sentence_embeddings = self._embeddings(contents, max_length=max_length)
        print("Sentence embeddings:", sentence_embeddings.shape)

        return sentence_embeddings

    def embedding_data(self, data, type='file', truncated_leagth=200, max_length=200):
        """
        对数据进行词嵌入
        :param data:
        :param type:
        :param truncated_leagth:
        :param max_length:
        :return:
        """
        if type == 'file':
            try:
                with open(data, 'r', encoding='utf-8') as f:
                    content = f.read()[:truncated_leagth]
            except UnicodeDecodeError:
                print(data)
        else:
            content = data[:truncated_leagth]
        sentence_embedding = self._embeddings(content, max_length=max_length)
        return sentence_embedding

    def _dataset_encode_utf8(self):
        dirs = os.listdir(self.dataset_path)
        for dir in dirs:
            dir_path = self.dataset_path + dir
            files = os.listdir(dir_path)
            for file in files:
                file_p = os.path.join(dir_path, file)
                try:
                    with open(file_p, 'r', encoding='utf-8') as f:
                        f.read()
                    os.remove(file_p)
                except UnicodeDecodeError:
                    # file_encode_ansi_2_uft8(file_p)
                    print(file_p)
                    file_encode_ansi_2_uft8(file_p)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)