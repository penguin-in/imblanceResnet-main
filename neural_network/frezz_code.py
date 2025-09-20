#
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import shutil
from model.resnet50mlp import SeedWeightvitalityPredictor,LSTMWeightPredictor
from fuction_file import seed_torch,WeightDataset,PureWeightDataset,PureEvaluate,train,ComputeMeanStd,train_lstm,evaluate_lstm,train_ablation,evaluate_ablation
from neural_network.model.resnet50mlp import AblationExperiment,SeedVitalityClassifier,vit
from fuction_file import train_simple_attention,evaluate_simple_attention
from torchvision.models import resnet101
#argparse############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--model_names",type=str,default="vision transformer")
parser.add_argument("--re_size",type=int,default="224")
parser.add_argument("--pre_trained",type=bool,default=False)
parser.add_argument("--classes_num",type=int,default=3)
parser.add_argument("--dataset",type=str,default="/neural_network/day0_image")
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--epoch",type=int,default=2500)
parser.add_argument("--lr",type=float,default=0.01)
parser.add_argument("--momentum",type=float,default=0.9)
parser.add_argument("--weight-decay",type=float,default=1e-5)
parser.add_argument("--seed",type=int,default=33)
parser.add_argument("--gpu-id",type=int,default=0)
parser.add_argument("--print_freq",type=int,default=1)
parser.add_argument("--exp_postfix",type=str,default="result_day0")
parser.add_argument("--txt_name",type=str,default="lr0.0001_wdSe-4")
parser.add_argument("--mode",type=int,default=4)
args = parser.parse_args()
image_dir = "/media/ls/办公/ls/seeddata/seeddata/day0_output"
output_dir = "day0_image"
train_ratio = 0.8
file_path = '/home/ls/code/seed/seedprosessing/sorted_output.xlsx'
train_dir = '/home/ls/code/seed/neural_network/day0_image/train'
test_dir = '/home/ls/code/seed/neural_network/day0_image/test'
#####################################################################################


all_sheets = pd.read_excel(file_path,sheet_name=None,header=None)
ori_data = np.concatenate((all_sheets['2'],all_sheets['3'],all_sheets['4'],
                       all_sheets['5'],all_sheets['6'],all_sheets['7'],all_sheets['8']
                    ,all_sheets['9'],all_sheets['10']),axis = 0)
data = np.concatenate((ori_data[:, 1:7], ori_data[:, 11].reshape(-1, 1)), axis=1)
data = np.array(data, dtype=np.float64)
OriginalWeight = data[:,2]
Day1Weight = data[:,3]
Day2Weight = data[:,4]
Day3Weight = data[:,5]
Vitality = data[:,6]

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".bmp"))]
image_files.sort(key=lambda x: int(re.search(r'\d+',x).group()))

if len(image_files) != len(Vitality):
    raise ValueError(f"image_files_len:{len(image_files)}labels_len{len(Vitality)}")

data_img = list(zip(image_files,OriginalWeight,Day1Weight,Day2Weight,Day3Weight,Vitality))
train_data,test_data = train_test_split(data_img,train_size=train_ratio,random_state=42)
def clear_output_dirs(subsets=None):
    if subsets is None:
        subsets = ["train", "test"]
    for subset in subsets:
        dir_path = os.path.join(output_dir, subset)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def copy_files(data, subset):
    for filename,OriginalWeight,Day1Weight,Day2Weight,Day3Weight,Vitality in data:
        src = os.path.join(image_dir, filename)
        dst_dir = os.path.join(output_dir, subset)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename)
        shutil.copy2(src, dst)

clear_output_dirs()
copy_files(train_data, "train")
copy_files(test_data, "test")

seed_torch(seed=args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
exp_name = args.exp_postfix
exp_path = "./model/{}/{}".format(args.model_names,exp_name)
os.makedirs(exp_path,exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transformer_train = transforms.Compose([
    #transforms.RandomRotation(90),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize((args.re_size,args.re_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.1274494230747223, 0.12850432097911835, 0.06643037497997284],[0.18800762295722961, 0.19520029425621033, 0.11122721433639526])
]
)
transformer_test = transforms.Compose([
    #transforms.RandomRotation(90),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize((args.re_size,args.re_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.1274494230747223, 0.12850432097911835, 0.06643037497997284],[0.18800762295722961, 0.19520029425621033, 0.11122721433639526])
]
)
if __name__ == "__main__":
    tb_path = "model/{}/{}".format(args.model_names,args.exp_postfix)
    BestVitalityModelPath = os.path.join(exp_path, "TestBestVitalityModel.pkl")
    tb_writer = SummaryWriter(log_dir=tb_path)
    save_path = os.path.join(exp_path,"SeedRegressionClasses","Best.pth")
    scaler_path = os.path.join(exp_path, "TestScaler.pkl")

    if args.mode == 0:
        #train_resnet50
        #model = SeedWeightvitalityPredictor()
        model = vit(num_classes=3)
        if torch.cuda.is_available():
            model = model.cuda()
        trainset = WeightDataset(train_dir, train_data, transform=transformer_train)
        testset = WeightDataset(test_dir, test_data, transform=transformer_train)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer,T_max=args.epoch)
        train(model, optimizer, train_loader, test_loader, scheduler, tb_writer, exp_path, args)

    elif args.mode == 1:
        #varify_resnet50
        #model = SeedWeightvitalityPredictor()
        model = vit(num_classes=3)
        if torch.cuda.is_available():
            model = model.cuda()
        pure_dir = "/mnt/d/pure_imag"
        PureImageFile = [f for f in os.listdir(pure_dir) if f.lower().endswith((".jpg", ".png", ".bmp"))]
        PureImageFile.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        TestData = all_sheets['11']
        TestData = TestData.to_numpy()
        TestData = np.concatenate((TestData[:, 1:7], TestData[:, 11].reshape(-1, 1)), axis=1)
        TestData = np.array(TestData, dtype=np.float64)
        TestDataOriWeight = TestData[:, 2]
        Day1WeightPure = TestData[:, 3]
        Day2WeightPure = TestData[:, 4]
        Day3WeightPure = TestData[:, 5]
        TestDataVitality = TestData[:, 6]
        if args.classes_num == 1:
            TargetWeight = Day1WeightPure
            PureData = list(zip(PureImageFile, TestDataOriWeight,TargetWeight))
            PureDataSet = PureWeightDataset(pure_dir, PureData, transform=transformer_train)
            PureDataLoader = DataLoader(PureDataSet, batch_size=args.batch_size, num_workers=0, shuffle=True)
            checkpoint = torch.load(save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(checkpoint['model'])
            model = model.cuda() if torch.cuda.is_available() else model
            model.eval()
            TestMae = PureEvaluate(model, PureDataLoader)
            print("TestMae:",TestMae)
        elif args.classes_num == 3:
            Day1WeightPure = torch.tensor(Day1WeightPure)
            Day2WeightPure = torch.tensor(Day2WeightPure)
            Day3WeightPure = torch.tensor(Day3WeightPure)
            TargetWeight = torch.stack([Day1WeightPure,Day2WeightPure,Day3WeightPure],dim=1)
            PureData = list(zip(PureImageFile, TestDataOriWeight, TargetWeight))
            PureDataSet = PureWeightDataset(pure_dir, PureData, transform=transformer_train)
            PureDataLoader = DataLoader(PureDataSet, batch_size=args.batch_size, num_workers=0, shuffle=True)
            checkpoint = torch.load(save_path,
                                    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(checkpoint['model'])
            model = model.cuda() if torch.cuda.is_available() else model
            model.eval()
            TestMae,Day1Mae,Day2Mae,Day3Mae = PureEvaluate(model, PureDataLoader,args.classes_num)
            print(f"TestMae:{TestMae} Day1Mae:{Day1Mae} Day2Mae:{Day2Mae} Day3Mae:{Day3Mae}")
    elif args.mode == 2:
        #train_lstm
        model = LSTMWeightPredictor()
        if torch.cuda.is_available():
            model = model.cuda()
        trainset = WeightDataset(train_dir, train_data, transform=transformer_train)
        testset = WeightDataset(test_dir, test_data, transform=transformer_train)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
        train_lstm(model, optimizer, train_loader, test_loader, scheduler, tb_writer, exp_path, args)
        #verify_lstm
        if torch.cuda.is_available():
            model = model.cuda()
        pure_dir = "/mnt/d/pure_imag"
        PureImageFile = [f for f in os.listdir(pure_dir) if f.lower().endswith((".jpg", ".png", ".bmp"))]
        PureImageFile.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        TestData = all_sheets['11']
        TestData = TestData.to_numpy()
        TestData = np.concatenate((TestData[:, 1:7], TestData[:, 11].reshape(-1, 1)), axis=1)
        TestData = np.array(TestData, dtype=np.float64)
        TestDataOriWeight = TestData[:, 2]
        Day1WeightPure = TestData[:, 3]
        Day2WeightPure = TestData[:, 4]
        Day3WeightPure = TestData[:, 5]
        TestDataVitality = TestData[:, 6]
        TargetWeight = Day3WeightPure
        PureData = list(zip(PureImageFile, TestDataOriWeight, Day1WeightPure,Day2WeightPure,Day3WeightPure,TestDataVitality))
        PureDataSet = WeightDataset(pure_dir, PureData, transform=transformer_test)
        PureDataLoader = DataLoader(PureDataSet, batch_size=args.batch_size, num_workers=0, shuffle=True)
        checkpoint = torch.load(save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model'])
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()
        VerifyMae = evaluate_lstm(model, PureDataLoader)
        print("VerifyMae:", VerifyMae)
    elif args.mode == 3:
        #train_abletion_experiment
        model = AblationExperiment()
        if torch.cuda.is_available():
            model = model.cuda()
        trainset = WeightDataset(train_dir, train_data, transform=transformer_train)
        testset = WeightDataset(test_dir, test_data, transform=transformer_train)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
        train_ablation(model, optimizer, train_loader, test_loader, scheduler, tb_writer, exp_path, args)
        # verify_ablation_experiment
        if torch.cuda.is_available():
            model = model.cuda()
        pure_dir = "/mnt/d/pure_imag"
        PureImageFile = [f for f in os.listdir(pure_dir) if f.lower().endswith((".jpg", ".png", ".bmp"))]
        PureImageFile.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        TestData = all_sheets['11']
        TestData = TestData.to_numpy()
        TestData = np.concatenate((TestData[:, 1:7], TestData[:, 11].reshape(-1, 1)), axis=1)
        TestData = np.array(TestData, dtype=np.float64)
        TestDataOriWeight = TestData[:, 2]
        Day1WeightPure = TestData[:, 3]
        Day2WeightPure = TestData[:, 4]
        Day3WeightPure = TestData[:, 5]
        TestDataVitality = TestData[:, 6]
        TargetWeight = Day3WeightPure
        PureData = list(
            zip(PureImageFile, TestDataOriWeight, Day1WeightPure, Day2WeightPure, Day3WeightPure, TestDataVitality))
        PureDataSet = WeightDataset(pure_dir, PureData, transform=transformer_test)
        PureDataLoader = DataLoader(PureDataSet, batch_size=args.batch_size, num_workers=0, shuffle=True)
        checkpoint = torch.load(save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model'])
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()
        _, Acc = evaluate_ablation(model, PureDataLoader)
        print("VerifyAcc:", Acc)
    elif args.mode == 4:
        #vision transformer
        model = resnet101(num_classes=1)
        if torch.cuda.is_available():
            model = model.cuda()
        trainset = WeightDataset(train_dir, train_data, transform=transformer_train)
        testset = WeightDataset(test_dir, test_data, transform=transformer_train)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
        train_simple_attention(model, optimizer, train_loader, test_loader, scheduler, tb_writer, exp_path, args)
        # verify_ablation_experiment
        if torch.cuda.is_available():
            model = model.cuda()
        pure_dir = "/media/ls/办公/ls/seeddata/seeddata/day0_varify"
        PureImageFile = [f for f in os.listdir(pure_dir) if f.lower().endswith((".jpg", ".png", ".bmp"))]
        PureImageFile.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        TestData = all_sheets['11']
        TestData = TestData.to_numpy()
        TestData = np.concatenate((TestData[:, 1:7], TestData[:, 11].reshape(-1, 1)), axis=1)
        TestData = np.array(TestData, dtype=np.float64)
        TestDataOriWeight = TestData[:, 2]
        Day1WeightPure = TestData[:, 3]
        Day2WeightPure = TestData[:, 4]
        Day3WeightPure = TestData[:, 5]
        TestDataVitality = TestData[:, 6]
        TargetWeight = Day3WeightPure
        PureData = list(
            zip(PureImageFile, TestDataOriWeight, Day1WeightPure, Day2WeightPure, Day3WeightPure, TestDataVitality))
        PureDataSet = WeightDataset(pure_dir, PureData, transform=transformer_test)
        PureDataLoader = DataLoader(PureDataSet, batch_size=args.batch_size, num_workers=0, shuffle=True)
        checkpoint = torch.load(save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model'])
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()
        _,Acc = evaluate_simple_attention(model, PureDataLoader)
        print("VerifyAcc:", Acc)













