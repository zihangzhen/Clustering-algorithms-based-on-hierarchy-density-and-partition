# Clustering-Algorithms-Based-on-Hierarchy-Density-and-Partition
Clustering Algorithms Based on Hierarchy, Density, and Partition. 
At the same time, it has also achieved the selection of initial values for kmeans and minibatch kmeans through the elbow method and Silhouette coefficient method


>- DBSCAAN:
>
>    基于密度的聚类方法，适用于非凸图像, args需要eps和min_samples
>
>- OPTICS:
>
>    基于密度的聚类方法，DBSCAAN的优化, 适用于非凸图像，args需要min_samples,xi和min_cluster_size
>
>- Agglomerative:
>
>    基于分层的聚类方法，适用于凸图像, args需要linkage, 包括ward,average,complete和single
>
>- KMeans:
>
>    基于划分的方法，适用于凸图像，简单有效, args需要n_clusters
>
>- MiniBatchKMeans:
>
>    基于划分的方法，优化KMeans计算过程，可分批训练，适用于凸图像，效果不如KMeans, args需要n_clusters,random_state和batch_size
>
>- ELBOW:
>
>    使用肘方法优化KMeans和的MiniBatchKMeans簇数选择, args需要k值的起、始值,步进值和消除抖动的幅值，使用MiniBatchKMean时还需要random_state和batch_size
