# eval.py
import argparse
import torch
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
import loaddataset
import net


def eval(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    eval_dataset = datasets.ImageFolder(root='dataset/val', transform=config.my_test_transform)  # 尺寸变化得到原图
    eval_data = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    model = net.SimCLRStage2(num_class=len(eval_dataset.classes)).to(DEVICE)
    model.load_state_dict(torch.load(config.pre_model_state2, map_location='cpu'), strict=False)  # 这个地方应该导入二阶段的训练结果

    # total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(eval_data)
    total_correct_1, total_correct_2, total_num = 0.0, 0.0, 0.0

    all_predictions, all_targets = [], []

    model.eval()
    with torch.no_grad():
        print("batch", " " * 1, "top1 acc", " " * 1, "top2 acc")
        for batch, (data, target) in enumerate(eval_data):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data)

            total_num += data.size(0)
            prediction = torch.argsort(pred, dim=-1, descending=True)
            top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            top2_acc = torch.sum((prediction[:, 0:2] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_1 += top1_acc
            total_correct_2 += top2_acc

            all_predictions.extend(prediction[:, 0:1].cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            print("  {:02}  ".format(batch + 1), " {:02.3f}%  ".format(top1_acc / data.size(0) * 100),
                  "{:02.3f}%  ".format(top2_acc / data.size(0) * 100))

        print("all eval dataset:", "top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100),
              "top2 acc:{:02.3f}%".format(total_correct_2 / total_num * 100))
        acc = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='macro')
        recall = recall_score(all_targets, all_predictions, average='macro')
        f1 = f1_score(all_targets, all_predictions, average='macro')

        print("all eval dataset:",
              "acc: {:02.3f}%".format(acc * 100),
              "precision: {:02.3f}%".format(precision * 100),
              "recall: {:02.3f}%".format(recall * 100),
              "f1: {:02.3f}".format(f1 * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test SimCLR')
    parser.add_argument('--batch_size', default=512, type=int, help='')

    args = parser.parse_args()
    eval(args)
