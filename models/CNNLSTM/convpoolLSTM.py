import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class conv_pooling(nn.Module):
    # Input size: 120x224x224
    # The CNN structure is first trained from single frame, then the FC layers are fine-tuned from scratch.
    def __init__(self, num_class):
        super(conv_pooling, self).__init__()

        self.conv = nn.Sequential(*list(torchvision.models.resnet18().children())[:-2])
        self.time_pooling = nn.MaxPool3d(kernel_size=(3, 1, 1))
        self.average_pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.linear1 = nn.Linear(1536, 2048)
        self.linear2 = nn.Linear(2048, num_class)

    def forward(self, x):
        # print("x shape", x.shape)
        t_len = x.size(2)
        # print("t_len", t_len)
        conv_out_list = []

        for i in range(t_len):
            conv_result = self.conv(torch.squeeze(x[:, :, i, :, :]))
            # print("conv_result", i, conv_result.shape)
            conv_out_list.append(conv_result)
            # print(i, conv_out_list)
        conv_stack = torch.stack(conv_out_list, 2)
        # print("0", conv_stack.shape)
        conv_out = self.time_pooling(conv_stack)
        # print("1", conv_out.shape)
        conv_out = self.average_pool(conv_out)
        # print("2", conv_out.shape)
        conv_out = self.linear1(conv_out.view(conv_out.size(0), -1))
        # print("3", conv_out.shape)
        conv_out = self.linear2(conv_out)
        # print("4", conv_out.shape)
        return conv_out


class cnn_lstm(nn.Module):
    # Input size: 30x224x224
    # The CNN structure is first trained from single frame, then the lstm is fine-tuned from scratch.
    def __init__(self, num_class):
        super(cnn_lstm, self).__init__()

        self.conv = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
        self.lstm = nn.LSTM(512, 512, 5, batch_first=True)
        self.fc = nn.Linear(512, num_class)

    # def forward(self, x):
    #     t_len = x.size(2)
    #     conv_out_list = []
    #     for i in range(t_len):
    #         conv_out_list.append(self.conv(torch.squeeze(x[:, :, i, :, :])))
    #     conv_out = torch.stack(conv_out_list, 1)
    #     conv_out, hidden = self.lstm(conv_out.view(conv_out.size(0), conv_out.size(1), -1))
    #     lstm_out = []
    #     for j in range(conv_out.size(1)):
    #         lstm_out.append(self.fc(torch.squeeze(conv_out[:, j, :])))
    #     conv_stack = torch.stack(lstm_out, 1)
    #     return conv_stack

    def forward(self, x_3d):
        conv_output_list = list()
        for t in range(x_3d.size(1)):
            conv_output_list.append(self.conv(x_3d[:, :, t, :, :]))
        conv_out = torch.stack(conv_output_list, 1)
        out, hidden = self.lstm(conv_out.view(conv_out.size(0), conv_out.size(1), -1))
        x = out[:, -1, :]
        x = F.relu(x)
        x = self.fc(x)
        return x


def conv_pooling_init():
    return conv_pooling(2)


def cnn_lstm_init():
    return cnn_lstm(2)
