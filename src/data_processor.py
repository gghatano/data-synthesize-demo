import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

def validate_data(df):
    """データの検証と警告の表示"""
    warnings = []
    errors = []
    
    # サイズチェック
    if len(df) > 100000:
        errors.append("データサイズが大きすぎます（最大100,000行まで）")
    
    # 欠損値チェック
    if df.isnull().any().any():
        warnings.append("欠損値が含まれています。自動的に処理されます。")
    
    # 数値型列の確認
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        errors.append("少なくとも2つの数値型列が必要です")
    
    return warnings, errors

def load_data(uploaded_file):
    """アップロードされたCSVファイルを読み込む"""
    if uploaded_file is not None:
        try:
            string_data = StringIO(uploaded_file.getvalue().decode('utf-8'))
            df = pd.read_csv(string_data)
            warnings, errors = validate_data(df)
            
            for warning in warnings:
                st.warning(warning)
            for error in errors:
                st.error(error)
                return None
                
            return df
        except Exception as e:
            st.error(f"データ読み込みエラー: {str(e)}")
            return None
    return None
