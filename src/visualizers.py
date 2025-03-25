import plotly.graph_objects as go

def plot_distribution_comparison(real_data, synthetic_data, column):
    """分布比較のプロット生成"""
    fig = go.Figure()
    
    # 実データのヒストグラム
    fig.add_trace(go.Histogram(
        x=real_data[column],
        name='Real Data',
        opacity=0.7,
        nbinsx=30,
        histnorm='probability'  # 相対頻度で表示
    ))
    
    # 合成データのヒストグラム
    fig.add_trace(go.Histogram(
        x=synthetic_data[column],
        name='Synthetic Data',
        opacity=0.7,
        nbinsx=30,
        histnorm='probability'  # 相対頻度で表示
    ))
    
    fig.update_layout(
        barmode='overlay',
        title=f'Distribution Comparison: {column}',
        xaxis_title=column,
        yaxis_title='Frequency'
    )
    
    return fig

def plot_correlation_matrices(real_corr, synthetic_corr, correlation_diff):
    """相関行列の比較プロット"""
    fig = go.Figure()
    
    # ボタンで切り替え可能な3つのヒートマップ
    figures = [
        ('Real Data', real_corr),
        ('Synthetic Data', synthetic_corr),
        ('Absolute Difference', correlation_diff)
    ]
    
    for i, (name, matrix) in enumerate(figures):
        fig.add_trace(go.Heatmap(
            z=matrix,
            x=matrix.columns,
            y=matrix.columns,
            colorscale='RdBu' if i < 2 else 'Reds',
            zmin=-1 if i < 2 else 0,
            zmax=1 if i < 2 else 1,
            name=name,
            visible=i == 0  # 最初は実データのみ表示
        ))
    
    # ボタンの設定
    buttons = []
    for i, (name, _) in enumerate(figures):
        visible = [j == i for j in range(len(figures))]
        buttons.append(dict(
            label=name,
            method="update",
            args=[{"visible": visible}]
        ))
    
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.15,
            showactive=True,
            buttons=buttons
        )],
        title="Correlation Matrix Comparison"
    )
    
    return fig
