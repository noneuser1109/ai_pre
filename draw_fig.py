import numpy as np
import plotly.graph_objects as go
import streamlit as st

def draw(t, learning_rate=0.05, x=0, y=0, epochs=50):
    if t == 0:
        # --- 1. 准备数据 ---
        X_data = np.array([1.0, 2.0, 3.0, 4.0])
        Y_data = np.array([3.0, 5.0, 7.0, 9.0])
        N = len(X_data)

        # --- 2. 定义辅助函数 ---

        def mse_loss(W, b, X, Y):
            """计算损失"""
            Y_pred = W * X + b
            loss = np.mean((Y_pred - Y) ** 2)
            return loss

        def compute_gradients(W, b, X, Y):
            """
            手动计算 MSE 关于 W 和 b 的梯度
            dL/dW = (2/N) * sum((y_pred - y) * x)
            dL/db = (2/N) * sum(y_pred - y)
            """
            Y_pred = W * X + b
            Y_pred = np.where(Y_pred >= 0, Y_pred, 0)
            error = Y_pred - Y

            grad_w = (2 / N) * np.sum(error * X)
            grad_b = (2 / N) * np.sum(error)
            return grad_w, grad_b

        # --- 3. 执行梯度下降 (Gradient Descent) ---

        # 超参数
        start_w = x  # 初始权重 (故意设得离最优值 W=2 远一点)
        start_b = y  # 初始偏置 (故意设得离最优值 b=1 远一点)

        # 记录历史数据用于画图
        w_history = []
        b_history = []
        loss_history = []

        # 初始化参数
        w_curr = start_w
        b_curr = start_b

        for i in range(epochs + 1):
            # 1. 计算当前损失并记录
            curr_loss = mse_loss(w_curr, b_curr, X_data, Y_data)
            w_history.append(w_curr)
            b_history.append(b_curr)
            loss_history.append(curr_loss)

            # 2. 计算梯度
            grad_w, grad_b = compute_gradients(w_curr, b_curr, X_data, Y_data)

            # 3. 更新参数 (theta_new = theta_old - lr * grad)
            w_curr = w_curr - learning_rate * grad_w
            b_curr = b_curr - learning_rate * grad_b

        # --- 4. 准备 3D 曲面数据 (背景) ---
        W_range = np.linspace(-1.0, 5.0, 50)
        b_range = np.linspace(-2.0, 4.0, 50)
        W_grid, B_grid = np.meshgrid(W_range, b_range)
        L_grid = np.zeros_like(W_grid)

        for i in range(W_grid.shape[0]):
            for j in range(W_grid.shape[1]):
                L_grid[i, j] = mse_loss(W_grid[i, j], B_grid[i, j], X_data, Y_data)

        # --- 5. 使用 Plotly 绘图 ---

        fig = go.Figure()

        # (A) 绘制半透明的损失曲面
        fig.add_trace(go.Surface(
            z=L_grid, x=W_grid, y=B_grid,
            colorscale='Viridis',
            opacity=0.6,  # 设置透明度，以便看清内部的线
            name='Loss Surface',
            showscale=False  # 隐藏颜色条以保持整洁
        ))

        # (B) 绘制梯度下降轨迹 (线 + 点)
        fig.add_trace(go.Scatter3d(
            x=w_history,
            y=b_history,
            z=loss_history,
            mode='lines+markers',  # 关键：把点连成线
            marker=dict(
                size=5,
                color='red',  # 点的颜色
                symbol='circle'
            ),
            line=dict(
                color='yellow',  # 线的颜色
                width=5
            ),
            name='Gradient Descent Path',
            text=[f'Epoch {i}<br>Loss: {l:.2f}' for i, l in enumerate(loss_history)]
        ))

        # (C) 标记起点 (Start)
        fig.add_trace(go.Scatter3d(
            x=[w_history[0]], y=[b_history[0]], z=[loss_history[0]],
            mode='text',
            text=['Start'],
            textposition="top center",
            textfont=dict(color='black', size=12)
        ))

        # (D) 标记终点/最优解 (Global Minimum)
        min_loss = mse_loss(2.0, 1.0, X_data, Y_data)
        fig.add_trace(go.Scatter3d(
            x=[2.0], y=[1.0], z=[min_loss],
            mode='markers',
            marker=dict(size=5, color='green', symbol='diamond'),
            name='True Minimum (2, 1)'
        ))

        # (E) 布局设置
        fig.update_layout(
            title='梯度下降可视化 (Gradient Descent Visualization)',
            scene=dict(
                xaxis_title='W (Weight)',
                yaxis_title='b (Bias)',
                zaxis_title='Loss',
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.2)  # 调整初始视角
                )
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40)
        )
    elif t == 1:
        # --- 1. 定义具有鞍点的损失函数 ---
        # 函数: L(x, y) = x^4 - 2x^2 + y^2
        # 鞍点在 (0, 0)，值为 0
        # 全局最小值在 (-1, 0) 和 (1, 0)，值为 -1

        def saddle_loss(x, y):
            return x ** 4 - 2 * x ** 2 + y ** 2

        def compute_gradients(x, y):
            """
            计算梯度:
            dL/dx = 4x^3 - 4x
            dL/dy = 2y
            """
            grad_x = 4 * x ** 3 - 4 * x
            grad_y = 2 * y
            return grad_x, grad_y

        # --- 2. 执行梯度下降 ---

        # 超参数设置
        # 关键策略：我们从高处开始，稍微偏离中心一点点 (x=0.01)。
        # 如果我们从严格的 x=0 开始，它将永远停在鞍点 (0,0)。
        # 偏离一点点可以让它先滑到鞍点附近，停滞一会儿，然后逃逸。
        start_x = x
        start_y = y
        learning_rate = learning_rate  # 使用较小的学习率以展示在鞍点附近的停滞
        epochs = epochs  # 需要更多的迭代次数来观察逃逸过程

        # 记录历史
        x_hist, y_hist, loss_hist = [], [], []
        x_curr, y_curr = start_x, start_y

        for i in range(epochs + 1):
            # 记录
            loss_hist.append(saddle_loss(x_curr, y_curr))
            x_hist.append(x_curr)
            y_hist.append(y_curr)

            # 计算梯度
            grad_x, grad_y = compute_gradients(x_curr, y_curr)

            # 更新 (标准 SGD)
            x_curr = x_curr - learning_rate * grad_x
            y_curr = y_curr - learning_rate * grad_y

        # --- 3. 准备 3D 曲面数据 (背景) ---
        # 创建网格范围
        range_val = 1.6
        X_grid, Y_grid = np.meshgrid(np.linspace(-range_val, range_val, 60),
                                     np.linspace(-range_val, range_val, 60))
        L_grid = saddle_loss(X_grid, Y_grid)

        # --- 4. Plotly 可视化 ---
        fig = go.Figure()

        # (A) 绘制半透明的损失曲面
        fig.add_trace(go.Surface(
            z=L_grid, x=X_grid, y=Y_grid,
            colorscale='RdBu_r',  # 使用红蓝反向色标，蓝色表示低谷，红色表示高峰
            opacity=0.7,
            name='Loss Surface',
            showscale=False
        ))

        # (B) 绘制梯度下降轨迹
        fig.add_trace(go.Scatter3d(
            x=x_hist, y=y_hist, z=loss_hist,
            mode='lines+markers',
            marker=dict(size=1, color='yellow', symbol='circle', opacity=0.8),
            line=dict(color='yellow', width=5),
            name='GD Path',
            text=[f'Epoch {i}' for i in range(len(x_hist))]
        ))

        # (C) 标记特殊点
        # 标记鞍点 (Saddle Point)
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers+text', marker=dict(size=12, color='red', symbol='diamond'),
            text=['Saddle Point (0,0)'], textposition="top center",
            name='Saddle Point'
        ))

        # 标记起点 (Start)
        fig.add_trace(go.Scatter3d(
            x=[x_hist[0]], y=[y_hist[0]], z=[loss_hist[0]],
            mode='text', text=['Start'], textposition="top center",
            textfont=dict(color='black', size=12)
        ))

        # 标记两个全局最小值 (Global Minima)
        fig.add_trace(go.Scatter3d(
            x=[-1, 1], y=[0, 0], z=[-1, -1],
            mode='markers', marker=dict(size=10, color='green', symbol='circle'),
            name='Global Minima'
        ))

        # (D) 布局设置
        fig.update_layout(
            title='梯度下降遇上鞍点 (Stuck at Saddle Point)',
            scene=dict(
                xaxis_title='X 参数',
                yaxis_title='Y 参数',
                zaxis_title='Loss',
                # 设置初始视角，更容易看清马鞍形状
                camera=dict(eye=dict(x=1.5, y=0.1, z=0.8)),
                # 调整 Z 轴的比例，让马鞍形状更明显
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=900, height=700,
            margin=dict(l=0, r=0, b=0, t=40)
        )
    return fig