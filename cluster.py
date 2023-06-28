#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Author   ：Zane
# @Mail     : zanezii@foxmail.com
# @Date     ：2023/6/22 21:23 
# @File     ：cluster.py
# @Description :

import sys
from tqdm import tqdm
from queue import Queue
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, MiniBatchKMeans, KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import random
from utils import serialization


class cluster:
    def __init__(self, path='./'):
        self.path = path
        self.test_X = self._init_test_data()
        self.model = None

    def _init_test_data(self):
        # matplotlib inline
        # X, y1 = datasets.make_circles(n_samples=5000, factor=.6, noise=.05)
        X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[2.5, 2.5]],
                                   cluster_std=[[.1]],
                                   random_state=9)
        for i in range(5):
            center_x = random.uniform(0, 5)
            center_y = random.uniform(0, 5)
            feature = random.randint(1, 4)
            sample = random.randrange(1000, 5500, 500)
            X1, y = datasets.make_blobs(n_samples=sample, n_features=feature, centers=[[center_x, center_y]],
                                        cluster_std=[[.1]],
                                        random_state=9)
            X = np.concatenate((X, X1))

        # 展示样本数据分布
        # plt.scatter(X[:, 0], X[:, 1], marker='o')
        # plt.show()
        return X

    def clustering(self, *args, method='KMeans', X=None, X_search=None, is_continue=False):
        """
        :param args:
        :param method:
            DBSCAAN:
                基于密度的聚类方法，适用于非凸图像,
                args需要eps和min_samples
            OPTICS:
                基于密度的聚类方法，DBSCAAN的优化，适用于非凸图像，
                args需要min_samples,xi和min_cluster_size
            Agglomerative:
                基于分层的聚类方法，适用于凸图像,
                args需要linkage, 包括ward,average,complete和single
            KMeans:
                基于划分的方法，适用于凸图像，简单有效，
                args需要n_clusters
            MiniBatchKMeans:
                基于划分的方法，优化KMeans计算过程，可分批训练，适用于凸图像，效果不如KMeans，
                args需要n_clusters,random_state和batch_size
            EBOW:
                使用肘方法优化KMeans和的MiniBatchKMeans簇数选择，
                args需要k值的起、始值,步进值和消除抖动的幅值，使用MiniBatchKMean时还需要random_state和batch_size
        :param X: 数据集，默认使用测试数据集
        :param X_search:
        :param is_continue: 是否继续训练模型，默认新训练模型
        :return:
        """
        if X is None:
            X = self.test_X
        print(f"-----------------{method}:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}---------------")

        if is_continue:
            # 继续训练模型
            if len(method)>=5 and (method[-5:] == '_mini' or method == 'MiniBatchKMeans'):
                new_args = [None, None, args[0]]
                model = self._mini_batch_kmeans(X,path=f'{self.path}{method}.joblib', *new_args)
            else:
                model = joblib.load(f'{self.path}{method}.joblib')
                model.fit(X)
        else:
            # 选择聚类方法，生成模型
            if method == 'DBSCAAN':
                if len(args) < 2:
                    print("[ERROR] DBSCAN need input parameters of eps and min_samples!")
                    sys.exit()
                model = DBSCAN(eps=args[0], min_samples=args[1])
            elif method == 'OPTICS':  # DBSCAAN的优化
                if len(args) < 3:
                    print("[ERROR] OPTICS need input parameters of min_samples, xi and min_cluster_size!")
                    sys.exit()
                model = OPTICS(min_samples=args[0], xi=args[1], min_cluster_size=args[2])
            elif method == 'Agglomerative':
                if len(args) < 1:
                    print("[ERROR] AgglomerativeClustering need input parameters of linkage!")
                    sys.exit()
                model = AgglomerativeClustering(linkage=args[0])
            elif method == 'KMeans':
                if len(args) < 1:
                    print("[ERROR] KMeans need input parameters of n_clusters!")
                    sys.exit()
                model = KMeans(n_clusters=args[0])
                model.fit(X)
            elif method == 'MiniBatchKMeans':
                if len(args) < 3:
                    print("[ERROR] MiniBatchKMeans need input parameters of n_clusters, random_state and batch_size!")
                    sys.exit()
                model = self._mini_batch_kmeans(X, None, *args)
            elif method == 'EBOW' or method== 'EBOW_mini':
                if len(args) < 3:
                    print("[ERROR] EBOW need input parameters of idnex of 'start' 'end', and maybe 'step' 'amplitude' and more!")
                    sys.exit()
                if args[0] == 'KMeans':
                    if len(args) < 5:
                        step = 1
                        amplitude = 1.0
                    else:
                        step = args[3]
                        amplitude = args[4]
                    model = self._ebow_kmeans(X=X, start=args[1], end=args[2], step=step, amplitude=amplitude)
                else:
                    if len(args) < 5:
                        step = 1
                        amplitude = 1.0
                        subargs = args[2:]
                    else:
                        step = args[3]
                        amplitude = args[4]
                        subargs = args[5:]
                    model = self._ebow_kmeans(X, args[1], args[2], step, amplitude, *subargs)
            elif method == 'Silhouette' or method== 'Silhouette_mini':
                if len(args) < 2:
                    print("[ERROR] Silhouette need input parameters of idnex of 'start' 'end', and more!")
                    sys.exit()
                model = self._silhouette_kmeans(X, X_search, *args)
            else:
                print(f"[ERROR] No such method {method}")
                sys.exit()

        # 保存模型，计算评价参数
        if method == 'DBSCAAN' or method == 'OPTICS' or method == 'Agglomerative':
            y_pred = model.fit_predict(X)
        else:
            y_pred = model.predict(X)

        joblib.dump(model, f'{self.path}{method}.joblib')

        if X_search is not None:
            try:
                y_search = model.predict(X_search)
            except:
                y_search = model.fit_predict(X_search)
            self._evalute_cluster_hit(y_pred, y_search, method, print_mode=False)
        self._evaluate_model(y_pred, X)
        self.model = model
        return model

    def _mini_batch_kmeans(self, X, path=None, *args):
        if path is None:
            model = MiniBatchKMeans(n_clusters=args[0], random_state=args[1], batch_size=args[2])
        else:
            model = joblib.load(f'{path}')

        batch_size = args[2]
        X_subs = [X[i:i + batch_size] if i + batch_size <= len(X) else X[i:len(X)] for i in
                  range(0, len(X), batch_size)]
        for X_sub in X_subs:
            model = model.partial_fit(X_sub)
        return model

    def _ebow_kmeans(self, X, start, end, step=1, amplitude=1.0, *args):
        is_minibatch = True if len(args) != 0 else False

        SSE = []
        SSE2 = []
        models = Queue()  # 通过在三阶函数判断，则最大时需要保存当前模型和前3个模型
        model = None
        index = 0
        for k in range(start, end, step):
            if is_minibatch:  # MiniBatchKMeans
                batch_size = args[1]
                kmeans_model = MiniBatchKMeans(n_clusters=k, random_state=args[0], batch_size=batch_size)
                X_subs = [X[i:i + batch_size] if i + batch_size <= len(X) else X[i:len(X)] for i in range(0, len(X), batch_size)]
                for X_sub in X_subs:
                    kmeans_model = kmeans_model.partial_fit(X_sub)
            else:  # KMeans
                kmeans_model = KMeans(n_clusters=k)
                kmeans_model.fit(X)
            SSE.append(kmeans_model.inertia_)  # 保存每一个k值的SSE值
            model = kmeans_model
            models.put(kmeans_model)
            # print('{} Means SSE loss = {}'.format(index, kmeans_model.inertia_))

            # 快速ebow,通过三阶最大值判断是否完成
            if index >= 2:
                # 当SSE计算到第[index]个时，计算第[index-2]个二阶SSE
                SSE2.append(SSE[index] + SSE[index - 2] - 2 * SSE[index - 1])
            if index >= 3:
                # 当SSE计算到第[index]个时，通过第[index-3]个三阶SSE，判断第[index-3]个二阶SSE，对应第[index-2]个K值
                models.get()
                # 默认递减函数
                left = (SSE[0] - SSE[index - 2])/((index-2)*step)
                right = SSE[index - 2] - SSE[index - 1]
                # print(left,right)
                if SSE2[index - 3] - SSE2[index - 2] > 0 and right / left < amplitude:
                    # 判断三阶右侧是否发生了转折，同时需要左右幅值变化够大以消除抖动
                    model = models.get()  # 返回第[index-2]个k的模型
                    print("[WARNING] The number of clusters selected by EBOW is", k - 2 * step)
                    index += 1
                    break
            index += 1

        # 绘制SSE图和二阶SSE图
        x = range(start, start + index * step, step)
        serialization(X, f'{self.path}_{is_minibatch}_x.pickle')
        serialization(SSE, f'{self.path}_{is_minibatch}_SSE.pickle')
        serialization(SSE2, f'{self.path}_{is_minibatch}_SSE2.pickle')
        self._plot_ebow_process_chart(x, SSE, SSE2)
        return model

    def _silhouette_kmeans(self, X, X_search=None, *args):
        start = args[0]
        end = args[1]
        step = args[2] if len(args) == 3 else 1
        is_minibatch = True if len(args) >= 5 else False
        method = 'silhouette_mini' if is_minibatch else 'silhouette'

        model = None
        scores = []
        hits = []
        best_score = -1
        K = None
        last_K = None
        for k in range(start, end, step):
            if is_minibatch:  # MiniBatchKMeans
                batch_size = args[4]
                kmeans_model = MiniBatchKMeans(n_clusters=k, random_state=args[3], batch_size=batch_size)
                X_subs = [X[i:i + batch_size] if i + batch_size <= len(X) else X[i:len(X)] for i in range(0, len(X), batch_size)]
                for X_sub in X_subs:
                    kmeans_model = kmeans_model.partial_fit(X_sub)
            else:  # KMeans
                kmeans_model = KMeans(n_clusters=k)
                kmeans_model.fit(X)
            Y = kmeans_model.predict(X)
            if X_search is not None: # 计算命中率
                Y_search = kmeans_model.predict(X_search)
                hits.append(self._evalute_cluster_hit(Y, Y_search, method=method))
            score = silhouette_score(X,Y)
            scores.append(score)  # 保存每一个轮廓系数
            if best_score <= score:
                model = kmeans_model
                best_score = score
                K = k
            last_K = k
        K = last_K if K==None else K
        print("[WARNING] The number of clusters selected by Silhouette is", K)
        # 绘制SSE图和二阶SSE图
        x = range(start, end, step)
        serialization(x, f'{self.path}_{is_minibatch}_Silhouette_x.pickle')
        serialization(scores, f'{self.path}__{is_minibatch}_Silhouette_scores.pickle')
        serialization(hits, f'{self.path}__{is_minibatch}_Silhouette_hits.pickle')
        self._plot_silhouette_process_chart(x, scores, hits)
        return model

    def _evaluate_model(self, y_pred, X=None):
        if X is None:
            X = self.test_X
            # 分类结果
            plt.scatter(X[:, 0], X[:, 1], c=y_pred)
            plt.show()

        try:
            # 以下几种评价指标对凸图像分数给的更高，对于DBSCAAN等不友好
            # 轮廓系数的取值范围在-1到1之间，越接近1表示聚类结果越好，越接近-1表示聚类结果越差，接近0则表示聚类结果存在重叠部分
            score = silhouette_score(X, y_pred)
            # calinski_harabasz分数越大表示聚类效果越好，因为它表示聚类之间的差异性相对于聚类内部的相似性更加明显。
            ch_score = metrics.calinski_harabasz_score(X, y_pred)
            # Davies-Bouldin分数越小表示聚类效果越好，因为它表示聚类之间的分离度相对于聚类内部的紧密度更加明显。
            # 比轮廓系数计算简单
            davies_score = davies_bouldin_score(X, y_pred)

            print("轮廓系数评分为(1)：", score)
            print("Calinski-Harabasz指数(↑)：", ch_score)
            print("Davies-Bouldin指数评分(↓)：", davies_score)
        except:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            pass

    def _plot_ebow_process_chart(self, x, SSE, SSE2):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        # 折线图观察最佳的k值
        ax1.plot(x, SSE, 'bx-')
        ax1.set_ylabel('SSE')
        ax1.set_xlabel('k')
        # 折线图观察二阶k值
        ax2.plot(x[1:-1], SSE2, 'bx-')
        ax2.set_ylabel('SSE2')
        ax2.set_xlabel('k')
        plt.show()

    def _plot_silhouette_process_chart(self, x, silhouette_scores, hits:list):
        if len(hits) == 0:
            plt.plot(x, silhouette_scores, 'bx-')
            plt.ylabel('silhouette score')
            plt.xlabel('k')
            plt.show()
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            # 折线图观察最佳的k值
            ax1.plot(x, silhouette_scores, 'bx-')
            ax1.set_ylabel('silhouette score')
            ax1.set_xlabel('k')
            # 折线图观察二阶k值
            ax2.plot(x, hits, 'bx-')
            ax2.set_ylabel('hit')
            ax2.set_xlabel('k')
            plt.show()

    def _evalute_cluster_hit(self, Y, Y_search, method, print_mode=True):
        method = '' if method is None else method
        hit = 0
        all = len(Y)
        for y, y_s in zip(Y, Y_search):
            hit = (hit + 1) if y == y_s else hit
        if print_mode:
            # print(f"\r{method}-{len(set(Y))}簇的命中率为： {hit/all*100}%", end="", flush=True)
            pass
        else:
            print(f"{method}-{len(set(Y))}簇的命中率为： {hit/all*100}%")
        return hit/all*100

    def evalute(self, X, X_search, method=None):
        model = self.model if method is None else joblib.load(f'{self.path}{method}.joblib')
        print(f"-----------------{method}:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}---------------")

        if method == 'DBSCAAN' or method == 'OPTICS' or method == 'Agglomerative':
            Y = model.fit_predict(X)
            Y_search = model.fit_predict(X_search)
        else:
            Y = model.predict(X)
            Y_search = model.predict(X_search)
        self._evalute_cluster_hit(Y,Y_search,method,print_mode=False)
        self._evaluate_model(Y,Y_search)


if __name__ == '__main__':
    c = cluster('./')
    # y_pred, _ = c.clustering(0.05, 5,method='DBSCAAN')
    # y_pred1, _ = c.clustering(50, 0.05, 0.05, method='OPTICS')
    # y_pred2, _ = c.clustering('single', method='Agglomerative')
    # y_pred3, _ = c.clustering(4, 0, 90, method='MiniBatchKMeans')
    # y_pred6, _ = c.clustering(4, method='KMeans')
    c.clustering('KMeans', 2, 10, 1, 0.35, method='EBOW')
    c.clustering('MiniBatchKMeans', 2, 10, 1, 0.35, 0, 90, method='EBOW')
    c.clustering(2, 10, 1, method='Silhouette')
    c.clustering(2, 10, 1, 0, 90, method='Silhouette')
