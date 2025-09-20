#Includes various types of functions
import os
import math
import cv2
import numpy as np
import torch
from PIL import Image
import re

from torch.utils.data import Dataset
import random
import time

from utils import AverageMeter,accuracy
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

def cosine_weights(epoch, total_epochs):
    weight_w = (math.cos(math.pi * epoch / total_epochs) + 1) / 2
    vitality_w = 1 - weight_w
    return vitality_w, weight_w

def seed_torch(seed=74):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class WeightDataset(Dataset):
    def __init__(self,image_dir,data_list,transform=None):
        self.image_dir = image_dir
        self.data_list = data_list
        self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_name,OriginalWeight_f,Day1Weight_f,Day2Weight_f,Day3Weight_f,Vitality_f = self.data_list[item]
        img_path = os.path.join(self.image_dir,img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        OriginalWeight_f = torch.tensor(OriginalWeight_f,dtype=torch.float32)
        Day1Weight_f = torch.tensor(Day1Weight_f,dtype=torch.float32)
        Day2Weight_f = torch.tensor(Day2Weight_f,dtype=torch.float32)
        Day3Weight_f = torch.tensor(Day3Weight_f,dtype=torch.float32)
        Vitality_f = torch.tensor(Vitality_f,dtype=torch.float32)
        match = re.search(r'\d+',img_name)
        if match:
            img_id = int(match.group())
        else:
            img_id = -1
        return img,OriginalWeight_f,Day1Weight_f,Day2Weight_f,Day3Weight_f,Vitality_f,img_id

class TestWeightDataset(Dataset):
    def __init__(self,image_dir,data_list,transform=None):
        self.image_dir = image_dir
        self.data_list = data_list
        self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_name,OriginalWeight_f = self.data_list[item]
        img_path = os.path.join(self.image_dir,img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        OriginalWeight_f = torch.tensor(OriginalWeight_f,dtype=torch.float32)
        match = re.search(r'\d+',img_name)
        if match:
            img_id = int(match.group())
        else:
            img_id = -1
        return img,OriginalWeight_f,img_id
class PureWeightDataset(Dataset):
    def __init__(self,image_dir,data_list,transform=None):
        self.image_dir = image_dir
        self.data_list = data_list
        self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_name,OriginalWeight_f,TestWeight = self.data_list[item]
        img_path = os.path.join(self.image_dir,img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        OriginalWeight_f = torch.tensor(OriginalWeight_f,dtype=torch.float32)
        TestWeight = torch.tensor(TestWeight,dtype=torch.float32)
        match = re.search(r'\d+',img_name)
        if match:
            img_id = int(match.group())
        else:
            img_id = -1
        return img,OriginalWeight_f,TestWeight,img_id

def ComputeMeanStd(image_dir):
    transform = transforms.ToTensor()

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    n_images = len(image_files)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for img_name in tqdm(image_files, desc="Computering:"):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)

        mean += img_tensor.mean(dim=(1, 2))
        std += img_tensor.std(dim=(1, 2))

    mean /= n_images
    std /= n_images
    print(f"image_mean: {mean.tolist()}, image_std: {std.tolist()}")
    return mean, std

def train_one_epoch(model,optimizer,train_loader,num):
    model.train()
    MaeRecoed = AverageMeter()
    MaeRecoed1 = AverageMeter()
    MaeRecoed2 = AverageMeter()
    MaeRecoed3 = AverageMeter()
    LossRecorder = AverageMeter()


    for image,OriginalWeight,Day1Weight,Day2Weight,Day3Weight,Vitality,img_id in tqdm(train_loader,desc='train'):
        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
            OriginalWeight = OriginalWeight.cuda(non_blocking=True)
            Day1Weight = Day1Weight.cuda(non_blocking=True)
            Day2Weight = Day2Weight.cuda(non_blocking=True)
            Day3Weight = Day3Weight.cuda(non_blocking=True)
            Vitality = Vitality.cuda(non_blocking=True)
        if num == 1:
            WeightPredict = model(image,OriginalWeight.unsqueeze(1))
            Mae = torch.mean(torch.abs(WeightPredict - Day1Weight))
            MaeRecoed.update(Mae.item(),n=image.size(0))
            optimizer.zero_grad()
            Mae.backward()
            optimizer.step()
        elif num==3:
            TrainWeight = torch.tensor([0.45, 0.45, 0.1], dtype=torch.float32, device=image.device).view(1, 3)
            WeightPredict = model(image, OriginalWeight)
            TargetWeight = torch.stack([Day1Weight, Day2Weight, Day3Weight], dim=1)
            AbsError = torch.abs(WeightPredict - TargetWeight)
            Mae = (AbsError * TrainWeight).sum(dim=1).mean()
            MaeRecoed.update(Mae.item(), n=image.size(0))
            optimizer.zero_grad()
            Mae.backward()
            optimizer.step()

    return MaeRecoed.avg

def evaluate(model,test_loader,num):
    model.eval()
    MaeRecoed = AverageMeter()

    with ((torch.no_grad())):
        for image, OriginalWeight, Day1Weight, Day2Weight, Day3Weight, Vitality,img_id in tqdm(test_loader, desc='Evaluating'):
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                OriginalWeight = OriginalWeight.cuda(non_blocking=True)
                Day1Weight = Day1Weight.cuda(non_blocking=True)
                Day2Weight = Day2Weight.cuda(non_blocking=True)
                Day3Weight = Day3Weight.cuda(non_blocking=True)
                Vitality = Vitality.cuda(non_blocking=True)

            if num == 1:
                WeightPredict = model(image, OriginalWeight.unsqueeze(1))
                Mae = torch.mean(torch.abs(WeightPredict - Day1Weight))
                MaeRecoed.update(Mae.item(), n=image.size(0))
            elif num == 3:
                EvaluateWeight = torch.tensor([0.45, 0.45, 0.1], dtype=torch.float32, device=image.device).view(1, 3)
                WeightPredict = model(image, OriginalWeight)
                TargetWeight = torch.stack([Day1Weight, Day2Weight, Day3Weight], dim=1)
                AbsError = torch.abs(WeightPredict - TargetWeight)
                Mae = (AbsError * EvaluateWeight).sum(dim=1).mean()
                MaeRecoed.update(Mae.item(), n=image.size(0))
    return MaeRecoed.avg

def PureEvaluate(model,test_loader,num):
    model.eval()
    MaeRecoed = AverageMeter()
    MaeRecoed1 = AverageMeter()
    MaeRecoed2 = AverageMeter()
    MaeRecoed3 = AverageMeter()
    if num == 1:
        with ((torch.no_grad())):
            for image,OriginalWeight,Weight,img_id in tqdm(test_loader, desc='Evaluating'):
                if torch.cuda.is_available():
                    image = image.cuda(non_blocking=True)
                    OriginalWeight = OriginalWeight.cuda(non_blocking=True)
                    Weight = Weight.cuda(non_blocking=True)

                    WeightPredict = model(image, OriginalWeight.unsqueeze(1))
                    Mae = torch.mean(torch.abs(WeightPredict - Weight))
                    MaeRecoed.update(Mae.item(), n=image.size(0))
        return MaeRecoed.avg
    elif num == 3:
        with ((torch.no_grad())):
            for image, OriginalWeight, Weight, img_id in tqdm(test_loader, desc='Evaluating'):
                if torch.cuda.is_available():
                    image = image.cuda(non_blocking=True)
                    OriginalWeight = OriginalWeight.cuda(non_blocking=True)
                    Weight = Weight.cuda(non_blocking=True)

                    WeightPredict = model(image, OriginalWeight)
                    Mae = torch.mean(torch.abs(WeightPredict - Weight))
                    MaeRecoed.update(Mae.item(), n=image.size(0))

                    MaeDay1 = torch.mean(torch.abs(WeightPredict[:,0]-Weight[:,0]))
                    MaeDay2 = torch.mean(torch.abs(WeightPredict[:,1]-Weight[:,1]))
                    MaeDay3 = torch.mean(torch.abs(WeightPredict[:,2]-Weight[:,2]))
                    MaeRecoed1.update(MaeDay1.item(),n=image.size(0))
                    MaeRecoed2.update(MaeDay2.item(),n=image.size(0))
                    MaeRecoed3.update(MaeDay3.item(),n=image.size(0))

        return MaeRecoed.avg,MaeRecoed1.avg,MaeRecoed2.avg,MaeRecoed3.avg

def train(model,optimizer,train_loader,test_loader,scheduler,tb_writer,exp_path,args):
    since = time.time()
    # max_acc = float('-inf')
    min_mae = float('inf')
    f = open(os.path.join(exp_path,"{}.txt".format(args.txt_name)),"w")

    for epoch in range(args.epoch):
        TrainMae = train_one_epoch(
              model,optimizer,train_loader,args.classes_num
        )
        print('TrainMae:',TrainMae)
        torch.cuda.empty_cache()
        TestMae = evaluate(model,test_loader,args.classes_num)
        print('TestMae:',TestMae)
        print('\n')
        if min_mae > TestMae:
            min_mae = TestMae
            print('min mae:',min_mae,'\n','epoch:',epoch,'\n')
            stat_dict = dict(epoch=epoch+1,model=model.state_dict(),min_mae=TestMae)
            name = os.path.join(exp_path,"SeedRegressionClasses","Best.pth")
            os.makedirs(os.path.dirname(name),exist_ok=True)
            torch.save(stat_dict,name)
        scheduler.step()

        tags = ['TrainMae',
                'TestMae'
                ]
        tb_writer.add_scalar(tags[0],TrainMae,epoch+1)
        tb_writer.add_scalar(tags[1],TestMae,epoch+1)
        if (epoch+1) % args.print_freq == 0:
            msg = ("epoch:{} model:{} TrainMae:{:.5f} TestMae:{:.5f}\n").format(epoch + 1, args.model_names, TrainMae, TestMae)
            print(msg)
            f.write(msg)
            f.flush()
    msg_best = "model:{} TestMae:{:.5f}\n".format(args.model_names,min_mae)
    time_elapsed = "trainning time:{}".format(time.time() - since)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()


def train_one_epoch_lstm(model,optimizer,train_loader):
    model.train()
    MaeRecoed = AverageMeter()

    for image,OriginalWeight,Day1Weight,Day2Weight,Day3Weight,Vitality,img_id in tqdm(train_loader,desc='train'):
        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
            OriginalWeight = OriginalWeight.cuda(non_blocking=True)
            Day1Weight = Day1Weight.cuda(non_blocking=True)
            Day2Weight = Day2Weight.cuda(non_blocking=True)
            Day3Weight = Day3Weight.cuda(non_blocking=True)

            InputWeight = torch.stack([OriginalWeight,Day1Weight, Day2Weight], dim=1)
            Day3Pre = model(image,InputWeight)
            Mae = torch.mean(torch.abs(Day3Pre - Day3Weight))
            MaeRecoed.update(Mae.item(), n=image.size(0))
            optimizer.zero_grad()
            Mae.backward()
            optimizer.step()

    return MaeRecoed.avg

def evaluate_lstm(model,test_loader):
    model.eval()
    MaeRecoed = AverageMeter()

    with ((torch.no_grad())):
        for image, OriginalWeight, Day1Weight, Day2Weight, Day3Weight, Vitality,img_id in tqdm(test_loader, desc='Evaluating'):
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                OriginalWeight = OriginalWeight.cuda(non_blocking=True)
                Day1Weight = Day1Weight.cuda(non_blocking=True)
                Day2Weight = Day2Weight.cuda(non_blocking=True)
                Day3Weight = Day3Weight.cuda(non_blocking=True)

                InputWeight = torch.stack([OriginalWeight, Day1Weight, Day2Weight], dim=1)
                Day3Pre = model(image, InputWeight)
                Mae = torch.mean(torch.abs(Day3Pre - Day3Weight))
                MaeRecoed.update(Mae.item(), n=image.size(0))

    return MaeRecoed.avg

def train_lstm(model,optimizer,train_loader,test_loader,scheduler,tb_writer,exp_path,args):
    since = time.time()
    # max_acc = float('-inf')
    min_mae = float('inf')
    f = open(os.path.join(exp_path,"{}.txt".format(args.txt_name)),"w")

    for epoch in range(args.epoch):
        TrainMae = train_one_epoch_lstm(
              model,optimizer,train_loader
        )
        print('TrainMae:',TrainMae)
        torch.cuda.empty_cache()
        TestMae = evaluate_lstm(model,test_loader)
        print('TestMae:',TestMae)
        print('\n')
        if min_mae > TestMae:
            min_mae = TestMae
            print('min mae:',min_mae,'\n','epoch:',epoch,'\n')
            stat_dict = dict(epoch=epoch+1,model=model.state_dict(),min_mae=TestMae)
            name = os.path.join(exp_path,"SeedRegressionClasses","Best.pth")
            os.makedirs(os.path.dirname(name),exist_ok=True)
            torch.save(stat_dict,name)
        scheduler.step()

        tags = ['TrainMae',
                'TestMae'
                ]
        tb_writer.add_scalar(tags[0],TrainMae,epoch+1)
        tb_writer.add_scalar(tags[1],TestMae,epoch+1)
        if (epoch+1) % args.print_freq == 0:
            msg = ("epoch:{} model:{} TrainMae:{:.5f} TestMae:{:.5f}\n").format(epoch + 1, args.model_names, TrainMae, TestMae)
            print(msg)
            f.write(msg)
            f.flush()
    msg_best = "model:{} TestMae:{:.5f}\n".format(args.model_names,min_mae)
    time_elapsed = "trainning time:{}".format(time.time() - since)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()

def train_one_epoch_ablation(model,optimizer,train_loader):
    model.train()
    LossRecorder = AverageMeter()
    AccRecorder = AverageMeter()
    LossFc = nn.BCEWithLogitsLoss()

    for _,OriginalWeight,Day1Weight,Day2Weight,Day3Weight,Vitality,img_id in tqdm(train_loader,desc='train'):
        if torch.cuda.is_available():
            OriginalWeight = OriginalWeight.cuda(non_blocking=True)
            Day1Weight = Day1Weight.cuda(non_blocking=True)
            Day2Weight = Day2Weight.cuda(non_blocking=True)
            Day3Weight = Day3Weight.cuda(non_blocking=True)
            Vitality = Vitality.cuda(non_blocking=True)

            InputWeight = torch.stack([OriginalWeight,Day1Weight, Day2Weight,Day3Weight], dim=1)
            out = model(InputWeight)
            loss = LossFc(out,Vitality.unsqueeze(1))
            LossRecorder.update(loss.item(), n=OriginalWeight.size(0))
            acc = accuracy(out,Vitality.unsqueeze(1))
            AccRecorder.update(acc,n=OriginalWeight.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return LossRecorder.avg,AccRecorder.avg

def evaluate_ablation(model,test_loader):
    model.eval()
    LossRecorder = AverageMeter()
    AccRecorder = AverageMeter()
    LossFc = nn.BCEWithLogitsLoss()

    with ((torch.no_grad())):
        for _, OriginalWeight, Day1Weight, Day2Weight, Day3Weight, Vitality,img_id in tqdm(test_loader, desc='Evaluating'):
            if torch.cuda.is_available():
                OriginalWeight = OriginalWeight.cuda(non_blocking=True)
                Day1Weight = Day1Weight.cuda(non_blocking=True)
                Day2Weight = Day2Weight.cuda(non_blocking=True)
                Day3Weight = Day3Weight.cuda(non_blocking=True)
                Vitality = Vitality.cuda(non_blocking=True)

                InputWeight = torch.stack([OriginalWeight, Day1Weight, Day2Weight,Day3Weight], dim=1)
                out = model(InputWeight)
                loss = LossFc(out, Vitality.unsqueeze(1))
                LossRecorder.update(loss.item(), n=OriginalWeight.size(0))
                acc = accuracy(out, Vitality.unsqueeze(1))
                AccRecorder.update(acc, n=OriginalWeight.size(0))


        return LossRecorder.avg, AccRecorder.avg

def train_ablation(model,optimizer,train_loader,test_loader,scheduler,tb_writer,exp_path,args):
    since = time.time()
    max_acc = float('-inf')
    f = open(os.path.join(exp_path, "{}.txt".format(args.txt_name)), "w")

    for epoch in range(args.epoch):
        TrainLoss, TrainAcc = train_one_epoch_ablation(
            model, optimizer, train_loader
        )
        print(f"TrainLoss:{TrainLoss:.4f} TrainAcc:{TrainAcc:.4f}%")
        torch.cuda.empty_cache()
        TestLoss, TestAcc = evaluate_ablation(model, test_loader)
        print(f"TestLoss:{TestLoss:.4f} TestAcc:{TestAcc:.4f}%")
        print('\n')
        if max_acc < TestAcc:
            max_acc = TestAcc
            print('max mcc:', max_acc, '\n', 'epoch:', epoch, '\n')
            stat_dict = dict(epoch=epoch + 1, model=model.state_dict(), max_acc=TestAcc)
            name = os.path.join(exp_path, "SeedRegressionClasses", "Best.pth")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            torch.save(stat_dict, name)
        scheduler.step()

        tags = ['TrainLoss',
                'TestLoss',
                'TrainAcc',
                'TestAcc'
                ]
        tb_writer.add_scalar(tags[0], TrainLoss, epoch + 1)
        tb_writer.add_scalar(tags[1], TestLoss, epoch + 1)
        tb_writer.add_scalar(tags[2], TrainAcc, epoch + 1)
        tb_writer.add_scalar(tags[3], TestAcc, epoch + 1)
        if (epoch + 1) % args.print_freq == 0:
            msg = ("epoch:{} model:{} TrainAcc:{:.5f}% TestAcc:{:.5f}%\n").format(epoch + 1, args.model_names, TrainAcc,
                                                                                  TestAcc)
            print(msg)
            f.write(msg)
            f.flush()
    msg_best = "model:{} TestMae:{:.5f}\n".format(args.model_names, max_acc)
    time_elapsed = "trainning time:{}".format(time.time() - since)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()




def loss_with_rejection_penalty(logits, targets, alpha=0.5):
    # mask = (targets != 2)
    # if mask.sum() > 0:
    #     main_loss = F.cross_entropy(logits[mask], targets[mask])
    # else:
    #     main_loss = torch.tensor(0.0, device=logits.device)
    #
    # probs = F.softmax(logits, dim=1)
    # rejection_probs = probs[:, 2]
    # reject_penalty = rejection_probs.mean()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor([0.5, 0.5], device=device)
    main_loss = F.cross_entropy(logits, targets,weight=weights)
    return main_loss

def focal_loss_binary(logits, targets, alpha=[0.7,0.3], gamma=1.0, reduction='mean'):
    B, C = logits.size()
    device = logits.device

    probs = F.softmax(logits, dim=1)  # [B, C]

    targets_onehot = F.one_hot(targets, num_classes=C).float()  # [B, C]
    p_t = (probs * targets_onehot).sum(dim=1)  # [B]

    if alpha is not None:
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
        alpha_t = (alpha * targets_onehot).sum(dim=1)  # [B]
    else:
        alpha_t = 1.0

    loss = - alpha_t * (1 - p_t).pow(gamma) * torch.log(p_t + 1e-8)  # [B]

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def focal_loss(logits,targets,alpha=[0.5,0.5],gamma=2.0,reduction='mean'):
    B,C = logits.size()
    device = logits.device

    probs = F.softmax(logits)
    targets_onehot = F.one_hot(targets,num_classes=C).float()
    p_t = (probs * targets_onehot).sum(dim=1)
    if alpha is not None:
        if isinstance(alpha,(list,tuple)):
            alpha = torch.tensor(alpha,device=device,dtype=torch.float32)
        alpha_t = (alpha*targets_onehot).sum(dim=1)
    else:
        alpha_t = (((1/C) * torch.ones(1,C))*targets_onehot).sum(dim=1)

    loss = -alpha_t*(1-p_t).pow(gamma)*torch.log(p_t+1e-8)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def train_one_epoch_3(model, optimizer, train_loader, device, alpha, gamma):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    pred_total = [0, 0, 0]
    acc_i = [0, 0, 0]

    for data in tqdm(train_loader, desc='train'):
        images, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = focal_loss_binary(outputs, labels,alpha=[1-alpha,alpha], gamma=gamma)
        # loss = loss_with_rejection_penalty(outputs, labels, alpha=0.5)
        loss.backward()
        optimizer.step() #参数更新

        total_loss += loss.item() * images.size(0)
        probs = F.softmax(outputs,dim=1)
        confidence,preds = torch.max(probs,dim=1)
        preds[confidence < 0.5] = 2
        # total_correct += (preds == labels).sum().item()
        # total_samples += images.size(0)

        for i in range(3):  # 假设3类：0，1，2
            class_total[i] += (labels == i).sum().item()
            class_correct[i] += ((preds == i) & (labels == i)).sum().item()
            pred_total[i] += (preds == i).sum().item()

    total_correct = class_correct[1] + class_correct[0]
    total_samples = class_total[0] + class_total[1]
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples * 100

    # print("\n=== Train Stats ===")
    evaluation_para = (
            class_correct[1]*(class_correct[1] / pred_total[1] -0.58)
    )if pred_total[1] > 0 else 0

    for i in range(3):
        acc_i[i] = 100 * class_correct[i] / pred_total[i] if pred_total[i] > 0 else 0
    #     print(f"Class {i}: Total={class_total[i]}, Predicted={pred_total[i]}, Acc={acc_i[i]:.2f}%")
    # print(f"the total correct num={class_correct[0] + class_correct[1]}")
    return avg_loss, acc,evaluation_para,acc_i[0],acc_i[1],acc_i[2],class_correct[0],class_correct[1],class_correct[2]

@torch.no_grad()
def evaluate_3(model, val_loader, device, alpha, gamma):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    pred_total = [0, 0, 0]
    acc_i = [0, 0, 0]
    for data in tqdm(val_loader, desc='eval'):
        images, labels = data[0].to(device), data[1].to(device)

        outputs = model(images)
        loss = focal_loss_binary(outputs, labels,alpha=[1-alpha,alpha], gamma=gamma)
        # loss = loss_with_rejection_penalty(outputs, labels, alpha=0.5)

        total_loss += loss.item() * images.size(0)
        probs = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, dim=1)
        preds[confidence < 0.5] = 2
        # total_correct += (preds == labels).sum().item()
        # total_samples += images.size(0)

        for i in range(3):  # 假设3类：0，1，2
            class_total[i] += (labels == i).sum().item()
            class_correct[i] += ((preds == i) & (labels == i)).sum().item()
            pred_total[i] += (preds == i).sum().item()

    total_correct = class_correct[1] + class_correct[0]
    total_samples = class_total[0] + class_total[1]
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples * 100
    evaluation_para = (
       class_correct[1]*(class_correct[1] / pred_total[1] -0.58)
    )if pred_total[1] > 0 else 0
    # print(f"\n=== test Stats === ")
    for i in range(3):
        acc_i[i] = 100 * class_correct[i] / pred_total[i] if pred_total[i] > 0 else 0
    #     print(f"Class {i}: Total={class_total[i]}, Predicted={pred_total[i]}, Acc={acc_i[i]:.2f}%")
    # print(f"the total correct num={class_correct[0]+class_correct[1]}")
    return avg_loss, acc, evaluation_para, acc_i[0], acc_i[1], acc_i[2], class_correct[0], class_correct[1], \
    class_correct[2]

def train_simple_3(model, optimizer, train_loader, test_loader, scheduler, exp_path, args,f):
    import time
    since = time.time()
    print(f"lr:{args.lr} alpha:{args.alpha} gamma:{args.gamma}")
    max_para = float('-inf')
    best_train_acc = 0
    best_train_class_correct2 = 0
    best_test_acc_2 = 0
    best_test_class_correct2 = 0
    for epoch in range(args.epoch):
        train_loss, train_acc,train_para,train_acc_1, train_acc_2, train_acc_3, train_class_correct1, train_class_correct2, \
        train_class_correct3= train_one_epoch_3(model, optimizer, train_loader, device=args.device, alpha = args.alpha,gamma = args.gamma)

        test_loss, test_acc,test_para ,test_acc_1, test_acc_2, test_acc_3, test_class_correct1, test_class_correct2, \
        test_class_correct3= evaluate_3(model, test_loader, device=args.device, alpha = args.alpha,gamma = args.gamma)

        if test_para > max_para:
            max_para = test_para
            best_train_acc = train_acc_2
            best_train_class_correct2 = train_class_correct2
            best_test_acc_2 = test_acc_2
            best_test_class_correct2 = test_class_correct2

            save_path = os.path.join(exp_path, "SeedRegressionClasses", "Best.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'lr':args.lr,
                'gamma':args.gamma,
                'alpha':args.alpha,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'max_acc': max_para
            }, save_path)

        scheduler.step()
    print(f"lr {args.lr} gamma {args.gamma} alpha {args.alpha} TrainAcc2: {best_train_acc:.4f}% TrainAcc2num: {best_train_class_correct2:.4f}"
               f"testAcc2: {best_test_acc_2:.4f}% testAcc2num: {best_test_class_correct2:.4f}\n")
    if max_para > 4:
        msg = (f"lr {args.lr} gamma {args.gamma} alpha {args.alpha} TrainAcc2: {best_train_acc:.4f}% TrainAcc2num: {best_train_class_correct2:.4f}"
               f"testAcc2: {best_test_acc_2:.4f}% testAcc2num: {best_test_class_correct2:.4f}\n")
        print(msg)
        f.write(msg)
        f.flush()
    time_elapsed = time.time() - since
    final_msg = f"Model: {args.model_names} Best para: {max_para:.4f}\nTraining time: {time_elapsed:.2f}s\n"
    print(final_msg)
    f.write(final_msg)
    return max_para

def compute_mean_std(image_dir, sample_count=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    imgs = os.listdir(image_dir)
    imgs = [f for f in imgs if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    if sample_count:
        imgs = imgs[:sample_count]

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for img_name in tqdm(imgs, desc=f"Processing {image_dir}"):
        img = Image.open(os.path.join(image_dir, img_name)).convert('RGB')
        img = transform(img)
        mean += img.mean(dim=(1, 2))
        std += img.std(dim=(1, 2))

    mean /= len(imgs)
    std /= len(imgs)
    return mean.tolist(), std.tolist()

class MultiImageDataset(Dataset):
    def __init__(self, data_list, image_dirs, transform_list):
        self.data_list = data_list
        self.image_dirs = image_dirs
        self.transform_list = transform_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_name, label = self.data_list[idx]
        imgs = []

        for i, dir_path in enumerate(self.image_dirs):
            img_path = os.path.join(dir_path, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transform_list[i]:
                img = self.transform_list[i](img)
            imgs.append(img)

        # Concatenate [3, H, W] × 6 => [18, H, W]
        input_tensor = torch.cat(imgs, dim=0)
        return input_tensor, int(label)