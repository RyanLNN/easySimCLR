# trainstage2.py
import torch, argparse, os
from torchvision import datasets
import loaddataset
import net, config
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# train stage two
def train(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(0))  # config.gpu_name
        # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    # load dataset for train and eval
    # train_dataset = CIFAR10(root='dataset', train=True, transform=config.train_transform, download=True)
    train_dataset = loaddataset.PreMyDatasetStage2(transform=config.my_test_transform)  # 这个地方不确定，增强可以增加鲁棒性
    # my_test_transform是针对原始图片进行调整，my_train_transform是对图片进行增强

    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # eval_dataset = CIFAR10(root='dataset', train=False, transform=config.test_transform, download=True)

    model = net.SimCLRStage2(num_class=len(train_dataset.classes)).to(DEVICE)
    model.load_state_dict(torch.load(args.pre_model, map_location='cpu'), strict=False)  # 加载预训练模型SimCLRStage1
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)  # 优化器只对fc层的参数进行修改

    os.makedirs(config.save_path, exist_ok=True)
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        total_loss = 0
        for batch, (data, target) in enumerate(train_data):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data)

            loss = loss_criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch ", epoch, "  batch ", batch, "  loss: ", loss.detach().item())

            total_loss += loss.item()

        print("epoch", epoch, "loss:", total_loss / len(train_dataset) * args.batch_size)
        with open(os.path.join(config.save_path, "stage2_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset) * args.batch_size) + " ")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, 'model_stage2_epoch' + str(epoch) + '.pth'))

            model.eval()
            with torch.no_grad():
                print("batch", " " * 1, "top1 acc", " " * 1, "top2 acc")
                total_loss, total_correct_1, total_correct_2, total_num = 0.0, 0.0, 0.0, 0
                for batch, (data, target) in enumerate(train_data):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    pred = model(data)

                    total_num += data.size(0)
                    prediction = torch.argsort(pred, dim=-1, descending=True)
                    top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top2_acc = torch.sum((prediction[:, 0:2] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_1 += top1_acc
                    total_correct_2 += top2_acc

                    print("  {:02}  ".format(batch + 1), " {:02.3f}%  ".format(top1_acc / data.size(0) * 100),
                          "{:02.3f}%  ".format(top2_acc / data.size(0) * 100))

                print("all eval dataset:", "top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100),
                      "top2 acc:{:02.3f}%".format(total_correct_2 / total_num * 100))
                with open(os.path.join(config.save_path, "stage2_top1_acc.txt"), "a") as f:
                    f.write(str(total_correct_1 / total_num * 100) + " ")
                with open(os.path.join(config.save_path, "stage2_top2_acc.txt"), "a") as f:
                    f.write(str(total_correct_2 / total_num * 100) + " ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=200, type=int, help='')
    parser.add_argument('--max_epoch', default=200, type=int, help='')
    parser.add_argument('--pre_model', default=config.pre_model, type=str, help='')

    args = parser.parse_args()
    train(args)
