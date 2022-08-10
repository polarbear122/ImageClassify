# 读取日志，并从中得到训练的损失和准确率，画出变化情况
from matplotlib import pyplot as plt


def read_txt(file_path):
    with open(file_path, encoding='utf-8') as file:
        content = file.read()
        print(content.rstrip())
        return content


# for line in content:
#     print(line)
if __name__ == "__main__":
    log_file = read_txt("./log_file.txt")
    log_file_split = log_file.split("\n")
    print(log_file_split)
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    _i = 0
    for log_line in log_file_split:
        epoch_loss_acc = log_line.split("|")
        epoch = int(epoch_loss_acc[0].split(":")[1])
        print("epoch", epoch)
        loss = float(epoch_loss_acc[1].split(":")[1])
        print("loss", loss)
        acc = float(epoch_loss_acc[2].split(":")[1].split("%")[0])
        print("acc", acc)
        print("log_line[29:39]:-%s-" % log_line[29:39])

        if log_line[29:39] == "TrainEpoch":
            train_loss.append(loss)
            train_acc.append(acc/100)
        else:
            test_loss.append(loss)
            test_acc.append(acc/100)
    print(train_loss, "\n", train_acc, "\n", test_loss, "\n", test_acc)
    plt.plot(train_loss, marker='o')
    plt.savefig("./log_img/train_loss.png")
    plt.close()
    plt.plot(train_acc, 'o:r')
    plt.savefig("./log_img/train_acc.png")
    plt.close()
    plt.plot(test_loss, marker='o')
    plt.savefig("./log_img/test_loss.png")
    plt.close()
    plt.plot(test_acc, 'o:r')
    plt.savefig("./log_img/test_acc.png")
    plt.close()
