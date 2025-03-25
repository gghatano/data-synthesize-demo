import numpy as np
import pandas as pd

def generate_test_data():
    """テストデータの生成（現実的な分布を使用）"""
    np.random.seed(42)
    n = 2000

    # 年齢を一様分布で生成（18歳以上64歳以下）
    age = np.random.uniform(18, 65, n)

    # 年齢と経験年数に相関を持たせる
    experience = np.maximum(0, age - 22 + np.random.normal(0, 2, n))

    # 収入を年齢と経験年数から計算
    base_income = 30000 + experience * 2000 + (age - 25) * 500
    income = base_income + np.random.normal(0, 5000, n)

    # 収入の下限を平滑化（25000付近の極端な集中を緩和）
    income = np.where(income < 25000,
                      25000 + np.random.uniform(0, 2000, size=n),
                      income)

    df = pd.DataFrame({
        'age': np.round(age).astype(int),
        'years_experience': np.round(experience).astype(int),
        'annual_income': np.round(income, -2).astype(int)  # 100単位で丸めて整数化
    })

    # 現実的な範囲にフィルタリング
    df = df[(df['age'] >= 18) & (df['age'] <= 64)]
    df = df[(df['annual_income'] >= 25000) & (df['annual_income'] <= 150000)]

    # データに微小なランダム性を追加してスムージング（整数化不要）
    return df
