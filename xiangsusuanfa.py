from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.datasets as datasets

# 数据加载
def load_cifar10(sample_size, augment, pca_components=128):   # 样本数量，随机翻转和裁剪，PCA降维数
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), #随机水平翻转图像
         transforms.RandomCrop(32, padding=4), #从图像中随机裁剪一个 32x32 的区域，并在周围填充 4 个像素
         transforms.ToTensor()] #将图像转换为 PyTorch 张量
        if augment #取决于augment
        else [transforms.ToTensor()]  #将图像转换为 PyTorch 张量
    )
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)   #加载训练集
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor()) #加载测试集

    X_train = train_dataset.data.reshape(-1, 32 * 32 * 3)
    X_test = test_dataset.data.reshape(-1, 32 * 32 * 3) # 将CIFAR-10 的图像数据从 (32, 32, 3) 格式转换为一维向量（展平）
    y_train = np.array(train_dataset.targets)
    y_test = np.array(test_dataset.targets)     #将数据集的标签提取到NumPy数组中

    # 从训练集中随机选取 sample_size 个样本用于训练
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # 对特征进行均值归零和方差归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) #计算训练数据的均值和标准差，使用计算出的均值和标准差对训练数据进行标准化，将标准化后的数据存储回去
    X_test = scaler.transform(X_test)

    # PCA 降维
    pca = PCA(n_components=pca_components) #创建一个 PCA 对象，参数为目标维数
    X_train = pca.fit_transform(X_train)  #计算训练数据的协方差矩阵，计算协方差矩阵的特征值和特征向量，使用特征向量的前 n_components 个来构建投影矩阵，将训练数据投影到目标维数空间，并返回
    X_test = pca.transform(X_test) #使用训练数据计算的投影矩阵对测试数据进行投影

    return X_train, X_test, y_train, y_test

# 模型训练和评估
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, is_last_iteration=False):
    model.fit(X_train, y_train) #使用训练集 X_train 和 y_train 拟合机器学习模型
    y_pred_test = model.predict(X_test) #使用模型对测试集X_test进行预测，并存储
    accuracy = accuracy_score(y_test, y_pred_test) #使用accuracy_score计算测试集上的准确率指标，并将结果存储在 accuracy 中
    print(f"Final Accuracy = {accuracy:.4f}")

    # 仅在最后一次输出分类报告和混淆矩阵
    if is_last_iteration:
        print(f"\n分类报告 - {model_name}")
        print(classification_report(y_test, y_pred_test)) #使用 sklearn.metrics.classification_report 函数打印分类报告
        plot_confusion_matrix(y_test, y_pred_test, classes=list(range(10)), model_name=model_name) #绘制混淆矩阵
    return accuracy

# 混淆矩阵可视化
def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# 主函数
def main():
    sample_sizes = [500 + 500 * i for i in range(20)]  # 每次增加 500 个样本，共 20 次
    results = {"KNN": [], "SVM": [], "Random Forest": []} # 存储各自的accuracy
    nmi_scores = []  # 存储 KMeans 的 NMI 分数

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),#要考虑的最近邻数
        "SVM": SVC(kernel='rbf', C=10, gamma=0.001),#径向基函数作为核函数，正则化参数，宽度参数
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=25, min_samples_split=5,#决策树数量，最大深度，拆分内部节点所需的最小样本数。
                                                min_samples_leaf=3, max_features="sqrt", random_state=12, n_jobs=-1),#叶节点所需的最小样本数，考虑的特征的平方根数量，随机数生成器的种子，使用所有可用的 CPU 内核来并行训练模型
    }

    for idx, size in enumerate(sample_sizes): #循环20次
        print(f"\n训练样本数量: {size}")
        X_train, X_test, y_train, y_test = load_cifar10(sample_size=size, augment=True, pca_components=100)
        is_last_iteration = (idx == len(sample_sizes) - 1)  # 判断是否是最后一次训练

        # 分类模型训练
        for model_name, model in models.items(): #循环3次，每次用model里不同的模型
            print(f"\n正在训练 {model_name} 模型，样本数量: {size}")
            accuracy = evaluate_model(model, X_train, X_test, y_train, y_test, model_name, is_last_iteration)
            results[model_name].append(accuracy)#将 accuracy 值添加到各自模型 results 字典中

        # KMeans 聚类
        print("\n正在评估 K-Means 聚类算法...")
        kmeans = KMeans(n_clusters=10, random_state=12, n_init=10) #创建的簇的数量，随机种子，初始化聚类中心点的次数为10
        y_clusters = kmeans.fit_predict(X_train) #将 K-Means 模型拟合到训练数据 X_train，并预测每个样本的簇标签，存储在 y_clusters 中
        nmi_score = normalized_mutual_info_score(y_train, y_clusters) #计算训练数据的真实标签
        nmi_scores.append(nmi_score) #将当前样本数量下的 NMI 分数添加到 nmi_scores 列表中
        print(f"当前样本数量 {size} 的 NMI 分数: {nmi_score:.4f}") #打印当前样本数量和相应的 NMI 分数
        if size==10000:
            print(f"\n分类报告 - K-means聚类算法")
            print(classification_report(y_train, y_clusters))  # 使用 sklearn.metrics.classification_report 函数打印分类报告
            plot_confusion_matrix(y_train, y_clusters, classes=list(range(10)), model_name="k-means")  # 绘制混淆矩阵

    # 绘制模型随样本数量变化的准确率曲线
    plt.figure(figsize=(10, 8))
    for model_name, acc_history in results.items():
        plt.plot(sample_sizes, acc_history, marker='o', label=model_name) #对于每个模型，绘制准确性历史记录与训练样本数量之间的折线图
    plt.title('Accuracy vs. Training Samples')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制 KMeans 的 NMI 变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, nmi_scores, marker='o', label='NMI') ##对于NMI，绘制准确性历史记录与训练样本数量之间的折线图
    plt.title('NMI vs. Training Samples (KMeans)')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('NMI Score')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
