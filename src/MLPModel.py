import json
import pickle

import numpy as np

import src.Layers as L

# 这个文件是一个多层感知机（MLP）模型的实现，包含了模型的构建、前向传播、反向传播、预测、深拷贝、保存和加载模型参数等功能。
# 该模型可以根据给定的网络结构和超参数进行初始化，并支持多种激活函数和优化器的使用。   
# 该模型还支持保存和加载模型参数，以便于在训练完成后进行推理或继续训练。
# 该模型的实现还包含了一个超参数搜索器（GridSearcher），用于自动化地搜索最佳的超参数组合，以提高模型的性能。   
# 该模型的实现还包含了一个训练器（Trainer），用于自动化地训练模型，并在每个epoch结束时评估模型的性能。
# 该模型的实现还包含了一个数据加载器（FashionMNISTDataloader），用于自动化地加载和预处理数据集，以便于模型的训练和评估。
class MLPModel:
    def __init__(self, nn_architecture=None):
        self.layers = []
        self.nn_architecture = nn_architecture if nn_architecture else []
        for layer in self.nn_architecture:
            self.add(L.Linear(layer["input_dim"], layer["output_dim"]))
            self.add(L.Activation(layer["activation"]))

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            if isinstance(layer, L.Linear):
                dW, db, grad = layer.backward(grad, layer.input_cache)
                layer.dW = dW
                layer.db = db
            elif isinstance(layer, L.Activation):
                grad = layer.backward(grad, layer.input_cache)
        return grad

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def deep_copy(self):
        """
        返回一份模型的深拷贝
        """
        model_copy = MLPModel(self.nn_architecture)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, L.Linear):
                model_copy.layers[i].W = layer.W.copy()
                model_copy.layers[i].b = layer.b.copy()
        return model_copy

    def save_model_dict(self, path):
        """
        保存模型参数和网络结构
        :param path: 保存路径（具体为 .pkl 文件的路径）
        """
        model_dict = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, L.Linear):
                model_dict[f"layer_{i}_W"] = layer.W
                model_dict[f"layer_{i}_b"] = layer.b
        with open(path, "wb") as f:
            pickle.dump(model_dict, f)
        with open(path.replace(".pkl", ".json"), "w") as f:
            json.dump(self.nn_architecture, f)

    def load_model_dict(self, path):
        """
        加载模型参数和网络结构
        :param path: 加载路径（具体为 .pkl 文件的路径）
        """
        with open(path.replace(".pkl", ".json"), "r") as f:
            nn_architecture = json.load(f)
        self.__init__(nn_architecture)
        with open(path, "rb") as f:
            model_dict = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, L.Linear):
                layer.W = model_dict[f"layer_{i}_W"]
                layer.b = model_dict[f"layer_{i}_b"]
                layer.zero_grad()
