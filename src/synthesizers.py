import streamlit as st
import time
import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def generate_synthetic_data(real_data, num_rows, method='gaussian_copula', epochs=100):
    """選択された手法で合成データを生成する（キャッシュ付き）"""
    try:
        # メタデータの作成
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_data)
        
        if method == 'gaussian_copula':
            model = GaussianCopulaSynthesizer(metadata)
            st.info('GaussianCopulaを使用してデータを生成中...')
            
            start_time = time.time()
            model.fit(real_data)
            synthetic_data = model.sample(num_rows)
            generation_time = time.time() - start_time
            
        elif method == 'BaseIndependent_Sampler':
            # SDV 1.17.3では独立サンプリングを手動で実装
            st.info('独立サンプリングモードを使用してデータを生成中...')
            start_time = time.time()
            
            # 各列を独立にサンプリング
            synthetic_data = pd.DataFrame()
            for column in real_data.columns:
                column_data = real_data[column]
                
                # 数値データの場合
                if np.issubdtype(column_data.dtype, np.number):
                    # 平均と標準偏差を計算
                    mean = column_data.mean()
                    std = column_data.std()
                    
                    # ガウス分布からサンプリング
                    sampled_values = np.random.normal(mean, std, num_rows)
                    
                    # 整数型の場合は四捨五入
                    if np.issubdtype(column_data.dtype, np.integer):
                        sampled_values = np.round(sampled_values).astype(int)
                        
                    # 最小・最大値を適用
                    min_val = column_data.min()
                    max_val = column_data.max()
                    sampled_values = np.clip(sampled_values, min_val, max_val)
                    
                # カテゴリデータの場合    
                else:
                    # 元のデータの確率分布に従ってサンプリング
                    value_counts = column_data.value_counts(normalize=True)
                    sampled_values = np.random.choice(
                        value_counts.index, 
                        size=num_rows, 
                        p=value_counts.values
                    )
                
                # 生成したデータをDataFrameに追加
                synthetic_data[column] = sampled_values
                
            generation_time = time.time() - start_time
            
        else:  # CTGAN
            model = CTGANSynthesizer(metadata, epochs=epochs)
            st.info(f'CTGANを使用してデータを生成中... (エポック数: {epochs})')
            
            start_time = time.time()
            model.fit(real_data)
            synthetic_data = model.sample(num_rows)
            generation_time = time.time() - start_time
        
        return synthetic_data, generation_time
        
    except Exception as e:
        st.error(f"データ生成エラー: {str(e)}")
        return None, 0
