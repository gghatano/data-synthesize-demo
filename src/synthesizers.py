import streamlit as st
import time
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
        elif method == 'BaseIndependent_Sampler':
            # BaseIndependentSamplerの代わりにGaussianCopulaを独立モードで使用
            model = GaussianCopulaSynthesizer(
                metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
                default_distribution='gaussian',
                # 独立サンプリングのために相関を無視する設定
                numerical_distributions={'copulas': 'gaussian'},
                correlation_method=None
            )
            st.info('独立サンプリングモードを使用してデータを生成中...')
        else:  # CTGAN
            model = CTGANSynthesizer(metadata, epochs=epochs)
            st.info(f'CTGANを使用してデータを生成中... (エポック数: {epochs})')
        
        start_time = time.time()
        
        # モデルの学習とサンプリング
        model.fit(real_data)
        synthetic_data = model.sample(num_rows)
        generation_time = time.time() - start_time
        
        return synthetic_data, generation_time
        
    except Exception as e:
        st.error(f"データ生成エラー: {str(e)}")
        return None, 0
