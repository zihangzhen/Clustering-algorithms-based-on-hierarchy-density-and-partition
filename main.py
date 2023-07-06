#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Author   ：Zane
# @Mail     : zaneii@foxmail.com
# @Date     ：2023/6/19 19:16 
# @File     ：main.py
# @Description :
import datetime
from textEmbeddings import textEmbeddings
from cluster import cluster
from utils import serialization, unserialization


def Embedding(path, truncated_leagth=200, max_length=200, order=True):
    tE = textEmbeddings(model=f'{path}bert_base')
    max_length = truncated_leagth if truncated_leagth > max_length else max_length
    X = tE.embedding_dataset(dataset=f'{path}dataset/20news-bydate-train/',
                             truncated_leagth=truncated_leagth,
                             max_length=max_length,
                             order=order)
    return X


def clustering(path, X, X_search):
    models = []
    c = cluster(f'{path}models/')
    c.clustering(0.05, 5, method='DBSCAAN', X=X, X_search=X_search)
    c.clustering(50, 0.05, 0.05, method='OPTICS', X=X, X_search=X_search)
    c.clustering('single', method='Agglomerative', X=X, X_search=X_search)
    c.clustering(20, 0, 1000, method='MiniBatchKMeans', X=X, X_search=X_search)
    c.clustering(20, method='KMeans', X=X, X_search=X_search)
    c.clustering('KMeans', 2, 41, 1, 0.35, method='ELBOW', X=X, X_search=X_search)
    c.clustering('MiniBatchKMeans', 2, 41, 1, 0.35, 0, 1000, method='ELBOW_mini', X=X, X_search=X_search)
    c.clustering(2, 41, 1, 0, 1000, method='Silhouette_mini', X=X, X_search=X_search)
    c.clustering(2, 41, 1, method='Silhouette', X=X, X_search=X_search)
    print(f"-----------------END:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}---------------")
    return models


def eval(X, X_search):
    c = cluster(f'{path}models/')
    c.evalute(method='DBSCAAN', X=X, X_search=X_search)
    c.evalute(method='OPTICS', X=X, X_search=X_search)
    c.evalute(method='Agglomerative', X=X, X_search=X_search)
    c.evalute(method='MiniBatchKMeans', X=X, X_search=X_search)
    c.evalute(method='KMeans', X=X, X_search=X_search)
    c.evalute(method='ELBOW', X=X, X_search=X_search)
    c.evalute(method='ELBOW_mini', X=X, X_search=X_search)
    c.evalute(method='Silhouette', X=X, X_search=X_search)
    c.evalute(method='Silhouette_mini', X=X, X_search=X_search)


if __name__ == '__main__':
    path = '/home/src/'
    order = False
    order_name = '_order' if order else '_disorder'
    leagth = 180

    # X = Embedding(path,order=False)
    # serialization(X,f'{path}dataset/20newsEmbeddings{order_name}.pickle')
    # X_search = Embedding(path,order=False,truncated_leagth=leagth,max_length=leagth)
    # serialization(X_search,f'{path}dataset/20newsEmbeddings_{leagth}{order_name}.pickle')

    X = unserialization(f'{path}dataset/20newsEmbeddings{order_name}.pickle')
    X_search = unserialization(f'{path}dataset/20newsEmbeddings_{leagth}{order_name}.pickle')
    clustering(path,X,X_search)

    # eval(X,X_search)