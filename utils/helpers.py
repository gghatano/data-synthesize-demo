import base64
import pandas as pd

def get_csv_download_link(df, filename="data.csv"):
    """DataFrameをダウンロードリンクとして提供"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href
