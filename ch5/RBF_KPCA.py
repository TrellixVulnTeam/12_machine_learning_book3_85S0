'''カーネル主成分分析 (新しい点のプロットに対応)'''

from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy import exp
import numpy as np


def rbf_kernel_pca2(X, gamma, n_components):
    # RBFカーネルPCAの実装
    # M × N次元のデータセットでペアごとのユークリッド距離の2乗を計算
    sq_dists = pdist(X, 'sqeuclidean')

    #ペアごとの距離を正方行列に変換
    mat_sq_dists = squareform(sq_dists)

    # 対称カーネル行列を計算
    K = exp(-gamma * mat_sq_dists)

    # カーネル行列を中心化
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - (one_n @ K) - (K @ one_n) + one_n @ K @ one_n

    # 中心化されたカーネル行列から固有対を取得
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # 上位k個の固有ベクトルを収集
    alphas = np.column_stack((eigvecs[:, i] for i in range(n_components)))

    # 対応する固有値を収集
    lambdas = [eigvals[i] for i in range(n_components)]

    return alphas, lambdas
