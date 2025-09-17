import numpy as np


def calculate_consistency(model_trends, ensemble_trend):
    num_models = model_trends.shape[0]
    consistency_mask = np.full(ensemble_trend.shape, False)
    for i in range(ensemble_trend.shape[0]):
        for j in range(ensemble_trend.shape[1]):
            # 计算与集合平均趋势符号一致的模式数量
            num_consistent_models = np.sum(np.sign(model_trends[:, i, j]) == np.sign(ensemble_trend[i, j]))
            # 判断是否超过80%
            if num_consistent_models / num_models >= 0.8:
                consistency_mask[i, j] = True
    return consistency_mask