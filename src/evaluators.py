import pandas as pd
from scipy.stats import ks_2samp

def calculate_distribution_similarity(real_data, synthetic_data, column):
    """KSテストを使用して分布の類似度を計算"""
    statistic, pvalue = ks_2samp(real_data[column], synthetic_data[column])
    return 1 - statistic  # 類似度に変換（1に近いほど類似）

def calculate_correlation_matrices(real_data, synthetic_data):
    """相関行列の計算と比較"""
    real_corr = real_data.corr()
    synthetic_corr = synthetic_data.corr()
    correlation_diff = abs(real_corr - synthetic_corr)
    return real_corr, synthetic_corr, correlation_diff
