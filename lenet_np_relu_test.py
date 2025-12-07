import numpy as np
import matplotlib.pyplot as plt
import torchvision

# -------------------------- 1. 加载.npz权重（纯NumPy） --------------------------
def load_weights_numpy(weight_path):
    """加载.npz权重，返回参数字典"""
    np_weights = np.load(weight_path)
    params = {key: np_weights[key].astype(np.float32) for key in np_weights.files}
    print(f"成功加载权重：{list(params.keys())}")
    return params

# -------------------------- 2. 加载MNIST测试集（纯NumPy，无需torchvision） --------------------------
def load_mnist_numpy():
    data_dir = './data'
    # 加载训练集
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=None  # 不使用 ToTensor()
    )

    X_train_np = train_dataset.data.float().numpy() / 255.0

    # 标签数据 (Y)
    Y_train_np = train_dataset.targets.numpy()

    # 使用 np.expand_dims 在索引 1 (Channel 维度) 处增加一个维度
    # 形状: (60000, 28, 28) -> (60000, 1, 28, 28)
    X_train_final = np.expand_dims(X_train_np, axis=1)
    return X_train_final, Y_train_np

# -------------------------- 3. 核心网络组件（纯NumPy） --------------------------
def relu(x):
    """ReLU激活函数"""
    return np.maximum(x, 0)

def conv2d_numpy(x, weight, bias, stride=1, padding=0):
    """纯NumPy实现2D卷积（适配(batch, in_c, h, w)输入）"""
    batch_size, in_c, h, w = x.shape
    out_c, _, k_h, k_w = weight.shape

    # 填充
    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')

    # 计算输出尺寸
    out_h = (h + 2*padding - k_h) // stride + 1
    out_w = (w + 2*padding - k_w) // stride + 1

    # 初始化输出
    output = np.zeros((batch_size, out_c, out_h, out_w), dtype=np.float32)

    # 卷积计算（批量优化，比四重循环快）
    for b in range(batch_size):
        for oc in range(out_c):
            # 单个卷积核
            kernel = weight[oc]  # (in_c, k_h, k_w)
            # 滑动窗口卷积
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    h_end = h_start + k_h
                    w_start = j * stride
                    w_end = w_start + k_w
                    # 点积 + 偏置
                    output[b, oc, i, j] = np.sum(x[b, :, h_start:h_end, w_start:w_end] * kernel) + bias[oc]
    return output

def maxpool2d_numpy(x, pool_size=2, stride=2):
    """纯NumPy实现最大池化"""
    batch_size, in_c, h, w = x.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    output = np.zeros((batch_size, in_c, out_h, out_w), dtype=np.float32)

    for b in range(batch_size):
        for c in range(in_c):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    h_end = h_start + pool_size
                    w_start = j * stride
                    w_end = w_start + pool_size
                    output[b, c, i, j] = np.max(x[b, c, h_start:h_end, w_start:w_end])
    return output

def forward_numpy(x, params):
    """纯NumPy前向传播"""
    # 卷积1 → ReLU → 池化
    conv1 = conv2d_numpy(x, params["c1_conv"], params["c1_b"], stride=1, padding=2)
    relu1 = relu(conv1)
    pool1 = maxpool2d_numpy(relu1, pool_size=2, stride=2)

    # 卷积2 → ReLU → 池化
    conv2 = conv2d_numpy(pool1, params["c3_conv"], params["c3_b"], stride=1, padding=0)
    relu2 = relu(conv2)
    pool2 = maxpool2d_numpy(relu2, pool_size=2, stride=2)

    # 展平
    batch_size = x.shape[0]
    flatten = pool2.reshape(batch_size, -1)  # (batch, 16*4*4=256)

    # 全连接层
    fc1 = np.dot(flatten, params["W1"]) + params["b1"]
    relu3 = relu(fc1)
    fc2 = np.dot(relu3, params["W2"]) + params["b2"]
    relu4 = relu(fc2)
    logits = np.dot(relu4, params["W3"]) + params["b3"]

    # 计算预测
    prob = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred = np.argmax(prob, axis=1)
    return prob, pred

# -------------------------- 4. 评估准确率（纯NumPy） --------------------------
def evaluate_numpy(params, x_test, y_test, epoch, batch_size=64):
    """批量评估（避免内存溢出）"""
    total_correct = 0
    total_samples = len(x_test)

    for i in range(0, total_samples, batch_size):
        batch_x = x_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        _, batch_pred = forward_numpy(batch_x, params)
        total_correct += np.sum(batch_pred == batch_y)
        print(f"已评估 {min(i+batch_size, total_samples)}/{total_samples} 样本")

    accuracy = 100 * total_correct / total_samples
    print(f"\nepoch:{epoch}纯NumPy模型测试集准确率：{accuracy:.2f}%")
    return accuracy

# -------------------------- 5. 单样本可视化（纯NumPy） --------------------------
def visualize_numpy_pred(params, x_test, y_test, idx=0):
    """可视化单个样本识别结果"""
    sample_x = x_test[idx:idx+1]
    true_label = y_test[idx]
    prob, pred_label = forward_numpy(sample_x, params)

    plt.figure(figsize=(6, 3))
    # 显示图像
    plt.subplot(1, 2, 1)
    plt.imshow(sample_x[0][0], cmap="gray")
    plt.title(f"True Label: {true_label}")
    plt.axis("off")
    # 显示概率
    plt.subplot(1, 2, 2)
    plt.bar(range(10), prob[0], color="lightblue")
    plt.xticks(range(10))
    plt.title(f"Pred Label: {pred_label[0]} (Prob: {prob[0][pred_label[0]]:.1f}%)")
    plt.ylabel("Probability")
    plt.show()

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 1. 加载权重（替换为你的.npz文件路径）
    for epoch in range(1, 16):
        weight_path = f"model_params_relu_epoch_{epoch}.npz"
        params = load_weights_numpy(weight_path)

        x_test, y_test = load_mnist_numpy()

        evaluate_numpy(params, x_test, y_test, epoch, batch_size=64)

        if epoch == 15: visualize_numpy_pred(params, x_test, y_test, idx=5)