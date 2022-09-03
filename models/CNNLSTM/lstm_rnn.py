import torch
import torch.nn as nn
import torch.nn.functional as F


class rnn_classify(nn.Module):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=2, num_layers=2):
        super(rnn_classify, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)  # 使用两层 lstm
        self.lstm = nn.LSTM(2048, 512, 5, batch_first=True)
        # batch_first:  If True, then the input and output tensors are provided as (batch, seq, feature). Default: False

        self.classifier = nn.Linear(hidden_feature, num_class)  # 将最后一个 rnn 的输出使用全连接得到最后的分类结果

    def forward(self, x):
        # x 大小为 (batch, 1, 28, 28)，所以我们需要将其转换成 RNN 的输入形式，即 (28, batch, 28)
        x = x.squeeze()  # 去掉 (batch, 1, 28, 28) 中的 1，变成 (batch, 28, 28)
        # x = x.permute(2, 0, 1)  # 将最后一维放到第一维，变成 (28, batch, 28)
        out, _ = self.rnn(x)  # 使用默认的隐藏状态，得到的 out 是 (28, batch, hidden_feature)
        out = out[-1, :, :]  # 取序列中的最后一个，大小是 (batch, hidden_feature)
        out = self.classifier(out)  # 得到分类结果
        return out


def rnn_classify_init():
    return rnn_classify()
