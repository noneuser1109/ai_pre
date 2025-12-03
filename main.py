import numpy as np
import plotly.graph_objects as go
import streamlit as st
from draw_fig import draw

@st.cache_data
def draw_data(t, lr, x, y, epochs):
    return draw(t, lr, x, y, epochs)

with st.sidebar:
    plot1 = st.button("y = wx + b")
    plot2 = st.button("l = x^4-2*x^2+y^2")

@st.fragment
def draw_fig1():
    col1, col2, col3, col4 = st.columns(4)
    learning_rate = col1.number_input(
        label='请输入学习率 (Learning Rate)',
        min_value=0.0001,
        max_value=1.0,
        value=0.05,
        step=0.001,
        format="%.4f",
        help='用于控制梯度下降的步长，建议值在 0.001 到 0.1 之间。',
        key='lr1'
    )

    # 2. 创建一个用于输入“迭代次数”的输入框
    epochs = st.number_input(
        label='请输入迭代次数 (Epochs)',
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help='梯度下降运行的轮数。',
        key='epochs1'
    )
    x = st.number_input(
        label='请输入w',
        min_value=-1.,
        max_value=5.,
        value=0.,
        step=0.01,
        help='权重',
        key='x1'
    )
    y = st.number_input(
        label='请输入b',
        min_value=-1.,
        max_value=5.,
        value=0.,
        step=0.01,
        help='偏置',
        key='y1'

    )
    if st.button("update"):
        st.plotly_chart(draw_data(0, learning_rate, x=x, y=y, epochs=epochs), use_container_width=True)

@st.fragment
def draw_fig2():
    col1, col2, col3, col4 = st.columns(4)
    learning_rate = col1.number_input(
        label='请输入学习率 (Learning Rate)',
        min_value=0.0001,
        max_value=1.0,
        value=0.05,
        step=0.001,
        format="%.4f",
        help='用于控制梯度下降的步长，建议值在 0.001 到 0.1 之间。',
        key='lr2'
    )

    # 2. 创建一个用于输入“迭代次数”的输入框
    epochs = st.number_input(
        label='请输入迭代次数 (Epochs)',
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help='梯度下降运行的轮数。',
        key='epochs2'
    )
    x = st.number_input(
        label='请输入x',
        min_value=-1.6,
        max_value=1.6,
        value=0.01,
        step=0.01,
        help='x',
        key='x2'
    )
    y = st.number_input(
        label='请输入y',
        min_value=-1.6,
        max_value=1.6,
        value=1.5,
        step=0.01,
        help='y',
        key='y2'
    )
    if st.button("update"):
        st.plotly_chart(draw_data(1, learning_rate, x=x, y=y, epochs=epochs), use_container_width=True)


if plot1:
    draw_fig1()
elif plot2:
    draw_fig2()