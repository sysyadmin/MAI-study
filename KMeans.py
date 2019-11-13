import numpy as np
import pandas as pd
import pylab as plt
import sklearn.cluster as km


class MyKMeans:
    """ Класс реализующий алгоритм К-Средних """
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels = None
        self.cluster_centers = None
        self.tol = 0.0001

    def fit(self, X):
        # Случайные центры
        self.cluster_centers = X[np.random.choice(range(X.shape[0]), self.n_clusters)].copy()

        samples = X.shape[0]
        self.labels = np.zeros(shape=(samples), dtype=np.uint8)
        min_dist = np.zeros(shape=(samples), dtype=np.float64)

        while True:
            # Перераспределение точек по кластерам
            for i in range(samples):
                min_dist[i] = np.linalg.norm(X[i] - self.cluster_centers[0])
                self.labels[i] = 0

            for clust in range(1, self.n_clusters):
                for i in range(samples):
                    dist = np.linalg.norm(X[i] - self.cluster_centers[clust])
                    if  dist < min_dist[i]:
                        min_dist[i], self.labels[i] = dist, clust
            # Пересчет центров
            new_centers = np.array([X[self.labels == i].sum(axis=0) / X[self.labels == i].shape[0] for i in range(self.n_clusters)])

            if (np.abs(new_centers - self.cluster_centers) < self.tol).all():
                break

            # print(min_dist, self.labels, self.cluster_centers, new_centers, sep='\n')
            # print("--------------------------------------------------------------\n")
            self.cluster_centers = new_centers.copy()

        return self

    def predict(self, X):
        samples = X.shape[0]
        min_dist = np.zeros(shape=(samples), dtype=np.float64)
        labels = np.zeros(shape=(samples), dtype=np.uint8)

        for i in range(samples):
                min_dist[i] = np.linalg.norm(X[i] - self.cluster_centers[0])
                labels[i] = 0

        for clust in range(1, self.n_clusters):
            for i in range(samples):
                dist = np.linalg.norm(X[i] - self.cluster_centers[clust])
                if  dist < min_dist[i]:
                    min_dist[i], labels[i] = dist, clust

        return labels


if __name__ == '__main__':
    # Простой пример, вроде работает
    # points = [
    #     [1, 2],
    #     [2, 1],
    #     [3, 1],
    #     [5, 4],
    #     [5, 5],
    #     [6, 5],
    #     [10, 8],
    #     [7, 9],
    #     [11, 5],
    #     [14, 9],
    #     [14, 14],
    # ]


    # k1 = KMeans(2).fit(np.array(points))
    # k2 = km.KMeans(2).fit(np.array(points))

    # print(k1.labels)
    # print(k2.labels_)
    
    data = pd.read_csv('MSFT.csv', index_col=0)
    # Нормировка Volume
    data['Volume'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
    print("Первые 10 записей MSFT.csv: ")
    print(data[:10])

    sklearn_kmeans = km.KMeans(3).fit(data.values)
    print("Центры, которые вычислила библиотека sklearn:")
    print(sklearn_kmeans.cluster_centers_)

    my_kmeans = MyKMeans(3).fit(data.values)
    print("Центры, которые вычислила MyKMeans:")
    print(my_kmeans.cluster_centers)

    plt.grid()
    plt.xlabel('High')
    plt.ylabel('Low')
    plt.scatter(data['High'],data['Low'], c=my_kmeans.labels)
    plt.show()

    print("Случайные 20 записей для пресказания: ")
    X = data.sample(20).values
    print(X)

    print("Предикт от sklearn: ")
    print(sklearn_kmeans.predict(X))
    
    print("Предикт от MyKMeans: ")
    print(my_kmeans.predict(X))
