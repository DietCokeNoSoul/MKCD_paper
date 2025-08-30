from torch import load, save, randn, flatten, split, cat, rand, matmul
from torch.nn import Module, Sequential, Linear, Sigmoid, ReLU, Tanh, Conv2d, MaxPool2d, LSTM, Parameter,Softmax
from torch.nn.functional import softmax, sigmoid, tanh


class Attention(Module):
    '''注意力机制'''

    def __init__(self, in_features: int, hiden_features: int, each_element: bool):
        super(Attention, self).__init__()
        if each_element:
            out_features = in_features
        else:
            out_features = 1
        self.features = Sequential(
            Linear(in_features, hiden_features),
            Tanh(),
            Linear(hiden_features, out_features),
            Sigmoid()
        )

    def forward(self, input):
        return self.features(input)*input


def make_layers(cfg: list, in_channels: int):
    '''根据配置列表cfg批量建立建立卷积层

    n->kernel_size=(2, 2), stride=1, padding=1

    M->MaxPool2d(kernel_size=2, stride=2, padding=1)

    Args:
        cfg:配置列表
        in_channel:初始输入通道数

    Return:
        Senuential对象
    '''
    layers = []
    for layer in cfg:
        if isinstance(layer, int):
            layers += [
                Conv2d(in_channels, layer, kernel_size=(
                    2, 2), stride=1, padding=1),
                ReLU(inplace=True)
            ]
            in_channels = layer
        else:
            layers += [
                MaxPool2d(kernel_size=2, stride=2, padding=1)
            ]
    return Sequential(*layers)


def out_shape(in_channels: int, in_features: int, cfg: list) -> int:
    '''计算多层卷积层后的特征图大小'''
    x = randn(size=(1, in_channels, 2, in_features))
    layer = make_layers(cfg, in_channels)
    x = layer(x)
    return x.shape


def get_full_linear(shape):
    '''计算下层全连接的网络输入大小'''
    return shape[-1]*shape[-2]*shape[-3]


def get_full_conv(shape):
    '''返回全卷积的大小(channels,kernal_size)'''
    return shape[1], (shape[2], shape[3])


class mSACN(Module):
    def __init__(self, in_channels: int, in_features: int, cfg: list):
        super(mSACN, self).__init__()
        self.features = make_layers(cfg, in_channels)
        in_shape = out_shape(in_channels, in_features, cfg)
        conv_channels, kernal_size = get_full_conv(in_shape)
        self.f_conv = Sequential(
            Conv2d(conv_channels, 2, kernal_size),
            Softmax(dim=1)
            # Sigmoid()
        )
        self._feature_length=get_full_linear(in_shape)

    def get_features(self,input):
        x=self.features(input)
        return x
    
    def get_result(self,x):
        x = self.f_conv(x)
        x = flatten(x, 1)
        return x

    def feature_length(self):
        return self._feature_length

    def forward(self, input):
        x=self.get_features(input)
        x=self.get_result(x)
        return x


class Attention_A1(Module):
    def __init__(self, in_features, hiden_features, views, scales):
        super(Attention_A1, self).__init__()
        self.W_v = Parameter(randn(1, 1, views, hiden_features, in_features))
        self.b_v = Parameter(randn(1, 1, views, hiden_features, 1))
        self.H = Linear(hiden_features, 1)

    def forward(self, input):
        x = input.view(*input.shape, 1)
        s_v = self.W_v.matmul(x).add(self.b_v).tanh().mean(dim=2)
        e_v = self.H(s_v.flatten(2)).softmax(dim=1)
        return input.mul(e_v.view(*e_v.shape, 1))


class Attention_A2(Module):
    def __init__(self, in_features, hiden_features, views, scales):
        super(Attention_A2, self).__init__()
        self.W_v = Parameter(randn(1, 1, views, hiden_features, in_features))
        self.b_v = Parameter(randn(1, 1, views, hiden_features, 1))
        self.w = Parameter(
            randn(1, scales, hiden_features, views*hiden_features))
        self.H = Linear(hiden_features, 1)

    def forward(self, input):
        x = input.view(*input.shape, 1)
        s_v = self.W_v.matmul(x).add(self.b_v).tanh()  # 16,4,3,50,1
        s_v = s_v.flatten(2)
        s_v = self.w.matmul(s_v.view(*s_v.shape, 1)).flatten(-2)
        e_v = self.H(s_v).softmax(dim=1)
        return input.mul(e_v.view(*e_v.shape, 1))


class Attention_B(Module):
    def __init__(self, in_features, hiden_features, scales):
        super(Attention_B, self).__init__()
        self.W = Parameter(randn(1, scales, hiden_features, in_features))
        self.b = Parameter(randn(1, scales, hiden_features, 1))
        self.H = Linear(hiden_features, 1)

    def forward(self, input):
        s = self.W.matmul(input.view(*input.shape, 1)
                          ).add(self.b).tanh().flatten(2)
        e = self.H(s).softmax(dim=1)
        return e


class RWRM(Module):
    def __init__(self, num_features: int, transfrom: int, views: int, steps: int):
        super(RWRM, self).__init__()
        self.transfrom = Linear(num_features, transfrom)
        self.attention_a = Attention_A2(transfrom, 50, views, steps+1)
        self.attention_b = Attention_B(transfrom, 50, steps+1)
        self.features = Sequential(
            LSTM(transfrom, 100,
                 bidirectional=True, batch_first=True),
        )
        self.classifier = Sequential(
            Linear(100*2*(steps+1), 2),
            # Sigmoid()
            Softmax(dim=1)
        )
        self.views = views
        self.scales = steps+1

    def get_features(self, input):
        x = self.transfrom(input)
        x = self.attention_a(x)
        x = x.sum(dim=-2)
        b = self.attention_b(x)
        x, _ = self.features(x)
        x = b.mul(x)
        x = flatten(x, 1)
        return x

    def feature_length(self):
        return 100*2*self.scales

    def get_result(self,x):
        x = self.classifier(x)
        return x

    def forward(self, input):
        x = self.get_features(input)
        x = self.classifier(x)
        return x


class MyModel(Module):
    def __init__(self,config_left,config_right) -> None:
        super(MyModel,self).__init__()
        self.model_left=None
        self.model_right=None
        if config_left is not None:
            self.model_left=RWRM(*config_left)
        if config_right is not None:
            self.model_right=mSACN(*config_right)
        self.integration=Sequential(
            Linear(self.model_left.feature_length()+self.model_right.feature_length(),2),
            Softmax(dim=1)
        )
    
    def left_result(self,x):
        d=self.model_left.get_features(x)
        y=self.model_left.get_result(d)
        return y

    def right_result(self,x):
        d=self.model_right.get_features(x)
        y=self.model_right.get_result(d)
        return y


    def forward(self,input_left,input_right):
        d_left=self.model_left.get_features(input_left)
        d_right=self.model_right.get_features(input_right)
        d=cat((d_left,d_right.view(-1,self.model_right.feature_length())),dim=-1)

        o_left=self.model_left.get_result(d_left)
        o_right=self.model_right.get_result(d_right)
        o=self.integration(d)
        return o,o_left,o_right
    
    def predict(self,input_left,input_right):
        d_left=self.model_left.get_features(input_left)
        d_right=self.model_right.get_features(input_right)
        d=cat((d_left,d_right.view(-1,self.model_right.feature_length())),dim=-1)
        o=self.integration(d)
        return o

