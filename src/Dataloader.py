import gzip
import numpy as np
import pickle

class CIFAR10Dataloader:
    def __init__(self, path_dir="cifar-10-batches-py", n_valid=1000, batch_size=32):
        # 初始化数据加载器
        self.path_dir = path_dir
        self.n_valid = n_valid
        self.batch_size = batch_size
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = self.load_data()

    def load_data(self):
        # 加载 CIFAR-10 数据
        X, y = self.load_cifar10_batches(self.path_dir, "data_batch")
        x_train, y_train, x_valid, y_valid = self.train_valid_split(X, y, self.n_valid)
        x_test, y_test = self.load_cifar10_batches(self.path_dir, "test_batch")
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    @staticmethod
    def load_cifar10_batches(path_dir, batch_name):
        data = []
        labels = []
        for i in range(1, 6) if batch_name == "data_batch" else [1]:
            filename = f"{path_dir}/{batch_name}_{i}" if batch_name == "data_batch" else f"{path_dir}/{batch_name}"
            with open(filename, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                data.append(batch['data'])
                labels.extend(batch['labels'])
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为 [N, 32, 32, 3] 格式
        data = data.astype(np.float32) / 255.0  # 归一化到 [0,1] 范围
        data = data.reshape(-1, 32 * 32 * 3)  # 展平为 [N, 3072] 格式
        labels = np.eye(10)[labels]  # 将标签转换为 one-hot 编码
        return data, labels

    @staticmethod
    def train_valid_split(x_train, y_train, n_valid):
        n_samples = x_train.shape[0]
        indices = np.random.permutation(n_samples)
        valid_indices = indices[:n_valid]
        train_indices = indices[n_valid:]
        return (x_train[train_indices], y_train[train_indices],
                x_train[valid_indices], y_train[valid_indices])

    def generate_train_batch(self):
        n_samples = self.x_train.shape[0]
        indices = np.random.permutation(self.x_train.shape[0])
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_train[batch_indices], self.y_train[batch_indices]

    def generate_valid_batch(self):
        n_samples = self.x_valid.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_valid[batch_indices], self.y_valid[batch_indices]

    def generate_test_batch(self):
        n_samples = self.x_test.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_test[batch_indices], self.y_test[batch_indices]