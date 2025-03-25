import streamlit as st
import pandas as pd
import numpy as np

# モジュールのインポート
from src.data_generator import generate_test_data
from src.data_processor import load_data, validate_data
from src.synthesizers import generate_synthetic_data
from src.evaluators import calculate_distribution_similarity, calculate_correlation_matrices
from src.visualizers import plot_distribution_comparison, plot_correlation_matrices
from utils.helpers import get_csv_download_link
from config.settings import CTGAN_EPOCH_INFO, MIN_EPOCH_COUNT, MAX_EPOCH_COUNT

# キャッシュデコレータをここに追加（元のコードでは関数に直接付与されていた）
@st.cache_data
def cached_generate_synthetic_data(real_data, num_rows, method='gaussian_copula', epochs=100):
    """generate_synthetic_data関数のキャッシュラッパー"""
    return generate_synthetic_data(real_data, num_rows, method, epochs)

def display_results():
    """評価結果の表示（セッションステートからデータを使用）"""
    if 'synthetic_data' not in st.session_state:
        return

    synthetic_data = st.session_state['synthetic_data']
    real_data = st.session_state['real_data']
    generation_time = st.session_state['generation_time']

    st.success(f'合成データの生成が完了しました！ (処理時間: {generation_time:.2f}秒)')
    
    # 合成データのダウンロードリンク
    st.markdown(
        get_csv_download_link(synthetic_data, "synthetic_data.csv"),
        unsafe_allow_html=True
    )
    
    # 評価結果の表示
    st.header('3. 評価結果')
    
    # 分布の比較
    st.subheader('3.1 分布の比較')
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns
    selected_column = st.selectbox('列を選択', numeric_columns)
    
    similarity = calculate_distribution_similarity(
        real_data, synthetic_data, selected_column
    )
    st.metric(
        "分布の類似度 (1に近いほど類似)",
        f"{similarity:.3f}"
    )
    
    fig = plot_distribution_comparison(
        real_data, synthetic_data, selected_column
    )
    st.plotly_chart(fig)
    
    # 相関行列の比較
    st.subheader('3.2 相関関係の比較')
    real_corr, synthetic_corr, correlation_diff = calculate_correlation_matrices(
        real_data, synthetic_data
    )
    
    mean_correlation_diff = correlation_diff.mean().mean()
    st.metric(
        "相関の平均差異 (0に近いほど類似)",
        f"{mean_correlation_diff:.3f}"
    )
    
    fig = plot_correlation_matrices(
        real_corr, synthetic_corr, correlation_diff
    )
    st.plotly_chart(fig)

def main():
    st.title('合成データ生成・評価デモ')
    
    # サイドバーにテストデータの説明を配置
    st.sidebar.header("テストデータについて")
    st.sidebar.write("""
    このデモアプリには、テスト用のサンプルデータが含まれています。
    - 1000行のデータ
    - 年齢、経験年数、年収の3つの列
    - 現実的な相関関係を含む
    """)
    
    # テストデータの生成とダウンロードリンク
    test_data = generate_test_data()
    st.sidebar.markdown(get_csv_download_link(test_data, "test_data.csv"), unsafe_allow_html=True)
    
    # コンテナを使用して各セクションを分離
    data_container = st.container()
    settings_container = st.container()
    results_container = st.container()
    
    with data_container:
        st.header('1. データの準備')
        data_input = st.radio(
            "データの入力方法を選択してください：",
            ['テストデータを使用', 'CSVファイルをアップロード']
        )
        
        real_data = None
        if data_input == 'テストデータを使用':
            real_data = test_data
            st.success("テストデータを読み込みました")
        else:
            uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])
            if uploaded_file is not None:
                real_data = load_data(uploaded_file)
        
        if real_data is not None:
            st.write("データの形状:", real_data.shape)
            st.write("サンプルデータ:", real_data.head())
    
    if real_data is not None:
        with settings_container:
            # 合成手法の選択
            st.header('2. 合成手法とパラメータの設定')
            col1, col2 = st.columns(2)
            
            with col1:
                method = st.selectbox(
                    '合成手法を選択',
                    ['BaseIndependent_Sampler', 'gaussian_copula', 'ctgan'],
                    format_func=lambda x: {
                        'BaseIndependent_Sampler': '独立サンプリング (高速・相関無視)',
                        'gaussian_copula': 'GaussianCopula (高速・相関の保持)',
                        'ctgan': 'CTGAN (高品質・低速)'
                    }[x]
                )
            
            with col2:
                num_rows = st.number_input(
                    '生成する行数',
                    min_value=1,
                    max_value=10000,
                    value=len(real_data)
                )
            
            # CTGANのパラメータ設定
            epochs = 50  # デフォルト値
            if method == 'ctgan':
                # エポック数を直接指定できるスライダーに変更
                epochs = st.slider(
                    'エポック数',
                    min_value=MIN_EPOCH_COUNT,
                    max_value=MAX_EPOCH_COUNT,
                    value=100,
                    format="%d",  # 整数値として表示
                )
                st.info(CTGAN_EPOCH_INFO)
            
            if st.button('合成データを生成', help='クリックするとデータ生成が始まります'):
                with st.spinner('データを生成中...'):
                    synthetic_data, generation_time = cached_generate_synthetic_data(
                        real_data, num_rows, method, epochs
                    )
                    
                    if synthetic_data is not None:
                        # セッションステートにデータを保存
                        st.session_state['synthetic_data'] = synthetic_data
                        st.session_state['real_data'] = real_data
                        st.session_state['generation_time'] = generation_time
    
    # 評価結果の表示（別のコンテナ内）
    with results_container:
        if 'synthetic_data' in st.session_state:
            display_results()

if __name__ == '__main__':
    main()
