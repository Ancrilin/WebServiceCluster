import numpy as np
from tqdm import tqdm
import time


class kmeans():
    def __init__(self, n_cluster=8, max_iter=300, tol=1e-4, init='k-means++'):
        self.n_cluster = n_cluster
        self.max_iter = max_iter                         #最大迭代次数
        self.tol = tol
        self.init = init
        self.centroid = None                             #质心
        self.label = None                                #聚类的标签

    def fit(self, X):                                    #训练kmeans++模型
        print("begin time", time.strftime('%m-%d %H.%M', time.localtime()))
        X = np.array(X)
        self.centroid = self._get_centroid(X)
        for i in tqdm(range(self.max_iter)):
            # if i % 10 == 0:
            #     print("iter:", i)
            assigned_sample = self._assign_sample_to_centroid(X, self.centroid)
            self.label = assigned_sample
            t_centriod, result = self._update_centroid(X, assigned_sample)
            # print("t_centriod", t_centriod)
            if result:
                break
            self.centroid = t_centriod
        print("end time", time.strftime('%m-%d %H.%M', time.localtime()))

    def predict(self, x):
        x = np.array(x)
        label = np.zeros(len(x), dtype=int)
        for i in range(len(x)):
            t_label = 0
            for j in range(self.n_cluster):
                if self._get_distane(X[i], self.centroid[j]) < self._get_distane(X[i], self.centroid[t_label]):
                    t_label = j
            label[i] = t_label
        return label

    #更新质心
    def _update_centroid(self, X, assigned_sample):
        t_centroid = np.zeros([self.n_cluster, len(X[0])], dtype=float)
        for i in range(self.n_cluster):
            t_centroid[i] = np.mean(X[np.where(assigned_sample==i)], axis=0)
        num = 0
        for i in range(self.n_cluster):
            if self._get_distane(t_centroid[i], self.centroid[i]) < self.tol:
                num = num + 1
        result = False
        if num == self.n_cluster:
            result = True
        return t_centroid, result

    #分类样本到最近的质心
    def _assign_sample_to_centroid(self, X, centroid):                        #分配样本到聚类中心
        assigned_sample = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            label = 0
            for j in range(self.n_cluster):
                if self._get_distane(X[i], centroid[j]) < self._get_distane(X[i], centroid[label]):
                    label = j
            assigned_sample[i] = label
        return assigned_sample

    # 计算质心
    def _get_centroid(self, X):
        centroid = np.zeros([self.n_cluster, len(X[0])], dtype=float)
        np.random.seed(int(time.time()))
        centroid[0] = X[np.random.randint(0, len(X))]   #第一个质心随机初始化
        for i in range(1, self.n_cluster):
            probability = self._get_sample_centroid_probability(X, i, centroid)
            flag = True
            while(flag):                                #防止出现重复的聚类中心
                pro = np.random.rand()
                index = -1
                for j in range(len(X)):
                    if pro <= probability[j]:
                        index = j
                        break
                if index != -1:
                    centroid[i] = X[index]
                    t_flag = True
                    for m in range(i):
                        if (centroid[i]==centroid[m]).all():
                            t_flag = False
                    if t_flag:
                        flag = False
        return centroid

    # 概率前缀和，获得前缀和后使用轮转法取得质心
    def _get_sample_centroid_probability(self, X, num, centroid):
        distance = np.zeros(len(X), dtype=float)
        sum_distance = 0
        for i in range(len(X)):
            d_distance = float('inf')
            for j in range(num):
                if self._get_distane(X[i], centroid[j]) < d_distance:
                    d_distance = self._get_distane(X[i], centroid[j])
                    if d_distance < 0:
                        d_distance = 0
            distance[i] = d_distance
            sum_distance = sum_distance + d_distance*d_distance
        probability = np.zeros(len(X), dtype=float)
        probability[0] = distance[0]*distance[0] / sum_distance
        for i in range(1, len(X)):
            t_probability = distance[i]*distance[i] / sum_distance
            probability[i] = probability[i-1] + t_probability
        return probability

    def _get_distane(self, vector1, vector2):            #余弦距离,距离越小越接近
        return 1 - np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))


if __name__ == '__main__':
    X = [[1,2,3,3], [2,8,0,5], [3,4,7,2], [9,56,7,2], [6,8,3,9]]
    kmeans = kmeans(n_cluster=3)
    kmeans.fit(X)
    print("label", kmeans.label)
    result = kmeans.predict([[3,5,2,4],[9,54,5,2]])
    print("centroid", kmeans.centroid)
    print(result)