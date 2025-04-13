import src.Layers as L


class SGDOptimizer:
    def __init__(self, lr, ld=0, decay_rate=0, decay_step=1000):
        self.lr = lr # 学习率，控制每次更新的步长
        self.ld = ld # L2正则化系数，控制模型复杂度
        self.decay_rate = decay_rate if 0 < decay_rate < 1 else None 
        # 学习率衰减率，控制学习率的衰减
        # 检查decay_rate范围
        self.decay_step = decay_step if 0 < decay_rate < 1 else None
        # 学习率衰减步数，指定每隔多少步更新一次学习率
        self.iterations = 0 # 迭代次数，用于学习率衰减的计算

    def step(self, model):
        for layer in model.layers:
            if isinstance(layer, L.Linear): # 检查层的类型
                # 更新权重和偏置
                layer.W -= self.lr * (layer.dW + self.ld * layer.W)
                layer.b -= self.lr * layer.db
                layer.zero_grad() # 清空梯度
        self.iterations += 1
        if self.decay_rate and self.iterations % self.decay_step == 0:
            self.lr *= self.decay_rate if self.lr > 0.01 else 1 # 更新学习率
