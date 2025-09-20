import argparse
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
from utils import AverageMeter,accuracy
import numpy as np
import time
import random
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import shutil
from torch.utils.data import Dataset
from model.resnet50mlp import SeedWeightVitalityPredictor3
import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
#argparse############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--model_names",type=str,default="resnet50")
parser.add_argument("--re_size",type=int,default="224")
parser.add_argument("--pre_trained",type=bool,default=False)
parser.add_argument("--classes_num",type=int,default=1)
parser.add_argument("--dataset",type=str,default="/neural_network/dataset_Reg")
parser.add_argument("--batch_size",type=int,default=16)
parser.add_argument("--epoch",type=int,default=100)
parser.add_argument("--lr",type=float,default=0.01)
parser.add_argument("--momentum",type=float,default=0.9)
parser.add_argument("--weight-decay",type=float,default=1e-5)
parser.add_argument("--seed",type=int,default=33)
parser.add_argument("--gpu-id",type=int,default=0)
parser.add_argument("--print_freq",type=int,default=1)
parser.add_argument("--exp_postfix",type=str,default="result1")
parser.add_argument("--txt_name",type=str,default="lr0.0002_wdSe-4")
parser.add_argument("--mode",type=int,default=0)
args = parser.parse_args()
image_dir = "/mnt/d/prosessed_imag"
output_dir = "dataset_reg"
train_ratio = 0.8
file_path = '/home/liushuai/seed/seedprosessing/sorted_output.xlsx'
train_dir = '/home/liushuai/seed/neural_network/dataset_reg/train'
test_dir = '/home/liushuai/seed/neural_network/dataset_reg/test'
#####################################################################################


all_sheets = pd.read_excel(file_path,sheet_name=None,header=None)

ori_data = np.concatenate((all_sheets['2'],all_sheets['3'],all_sheets['4'],
                       all_sheets['5'],all_sheets['6'],all_sheets['7'],all_sheets['8']
                    ,all_sheets['9'],all_sheets['10'],all_sheets['11']),axis = 0)
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


###########seed################
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
    transforms.Normalize((0.3738,0.3738,0.3738),(0.3240,0.3240,0.3240))
]
)
transformer_test = transforms.Compose([
    #transforms.RandomRotation(90),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize((args.re_size,args.re_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.3738,0.3738,0.3738),(0.3240,0.3240,0.3240))
]
)
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


AllDataset = WeightDataset(image_dir, data_img, transform=transformer_train)


AllDataLoader = DataLoader(AllDataset,batch_size=args.batch_size,num_workers=0,shuffle=True)


def train_one_epoch(model,optimizer,train_loader,vitality_w, weight_w):
    model.train()
    Bce_Loss = nn.BCEWithLogitsLoss()
    MSE_Loss = nn.MSELoss()
    MaeRecoed = AverageMeter()
    MseRecord = AverageMeter()
    AccRecorder = AverageMeter()
    FcrossRecorder = AverageMeter()
    LossRecorder = AverageMeter()

    for image,OriginalWeight,Day1Weight,Day2Weight,Day3Weight,Vitality,img_id in tqdm(train_loader,desc='train'):
        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
            OriginalWeight = OriginalWeight.cuda(non_blocking=True)
            Day1Weight = Day1Weight.cuda(non_blocking=True)
            Day2Weight = Day2Weight.cuda(non_blocking=True)
            Day3Weight = Day3Weight.cuda(non_blocking=True)
            Vitality = Vitality.cuda(non_blocking=True)

        VitalityPredict,WeightPredict = model(image,OriginalWeight.unsqueeze(1))
        # Vitality = Vitality.clone().detach().float()
        # Vitality = Vitality.unsqueeze(1)
        # VitalityLoss = Bce_Loss(VitalityPredict,Vitality)
        TargetWeight = torch.stack([Day1Weight, Day2Weight, Day3Weight], dim=1)
        # WeightLoss = MSE_Loss(WeightPredict, TargetWeight)
        # FcrossRecorder.update(VitalityLoss.item(),n=image.size(0))
        # MseRecord.update(WeightLoss.item(),n=image.size(0))
        # acc = accuracy(VitalityPredict, Vitality)
        # AccRecorder.update(acc, n=image.size(0))
        Mae = torch.mean(torch.abs(WeightPredict - TargetWeight))
        MaeRecoed.update(Mae.item(),n=image.size(0))
        optimizer.zero_grad()
        Mae.backward()
        optimizer.step()
    return LossRecorder.avg,AccRecorder.avg,FcrossRecorder.avg,MseRecord.avg,MaeRecoed.avg

def evaluate(model,test_loader,vitality_w,weight_w):
    model.eval()
    Bce_Loss = nn.BCEWithLogitsLoss()
    MSE_Loss = nn.MSELoss()
    MaeRecoed = AverageMeter()
    MseRecord = AverageMeter()
    AccRecorder = AverageMeter()
    FcrossRecorder = AverageMeter()
    LossRecorder = AverageMeter()

    with ((torch.no_grad())):
        for image, OriginalWeight, Day1Weight, Day2Weight, Day3Weight, Vitality,img_id in tqdm(test_loader, desc='Evaluating'):
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                OriginalWeight = OriginalWeight.cuda(non_blocking=True)
                Day1Weight = Day1Weight.cuda(non_blocking=True)
                Day2Weight = Day2Weight.cuda(non_blocking=True)
                Day3Weight = Day3Weight.cuda(non_blocking=True)
                Vitality = Vitality.cuda(non_blocking=True)

            VitalityPredict, WeightPredict = model(image, OriginalWeight.unsqueeze(1))
            Vitality = Vitality.clone().detach().float()
            Vitality = Vitality.unsqueeze(1)
            # print('result:',VitalityPredict.squeeze(1))
            # print('real:',Vitality)
            # VitalityLoss = Bce_Loss(VitalityPredict, Vitality)
            TargetWeight = torch.stack([Day1Weight, Day2Weight, Day3Weight], dim=1)
            WeightLoss = MSE_Loss(WeightPredict, TargetWeight)
            # FcrossRecorder.update(VitalityLoss.item(), n=image.size(0))
            MseRecord.update(WeightLoss.item(), n=image.size(0))
            acc = accuracy(VitalityPredict, Vitality)
            AccRecorder.update(acc, n=image.size(0))
            Mae = torch.mean(torch.abs(WeightPredict - TargetWeight))
            MaeRecoed.update(Mae.item(), n=image.size(0))
            loss = Mae
            LossRecorder.update(loss.item(), n=image.size(0))

    return loss, AccRecorder.avg, FcrossRecorder.avg, MseRecord.avg, MaeRecoed.avg

def train(model,optimizer,train_loader,test_loader,scheduler,tb_writer):
    since = time.time()
    # max_acc = float('-inf')
    min_mae = float('inf')
    f = open(os.path.join(exp_path,"{}.txt".format(args.txt_name)),"w")

    for epoch in range(args.epoch):
        vitality_w,weight_w = cosine_weights(epoch,args.epoch)
        TrainLoss,TrainAcc,TrainFcross,TrainMse,TrainMae = train_one_epoch(
              model,optimizer,train_loader,vitality_w,weight_w
        )
        print('TrainLoss:',TrainLoss)
        print('TrainAcc:',TrainAcc)
        print('TrainFcross:',TrainFcross)
        print('TrainMse:',TrainMse)
        print('TrainMae:',TrainMae)
        print('\n')
        torch.cuda.empty_cache()
        TestLoss,TestAcc,TestFcross,TestMse,TestMae = evaluate(model,test_loader,vitality_w,weight_w)
        print('TestLoss:',TestLoss)
        print('TestAcc:',TestAcc)
        print('TestFcross:',TestFcross)
        print('TestMse:',TestMse)
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

        tags = ['TrainLoss',
                'TrainAcc',
                'TrainFcross',
                'TrainMse',
                'TrainMae',
                'TestLoss',
                'TestAcc',
                'TestFcross',
                'TestMse',
                'TestMae'
                ]
        tb_writer.add_scalar(tags[0],TrainLoss,epoch+1)
        tb_writer.add_scalar(tags[1],TrainAcc,epoch+1)
        tb_writer.add_scalar(tags[2],TrainFcross,epoch+1)
        tb_writer.add_scalar(tags[3],TrainMse,epoch+1)
        tb_writer.add_scalar(tags[4],TrainMae,epoch+1)
        tb_writer.add_scalar(tags[5],TestLoss,epoch+1)
        tb_writer.add_scalar(tags[6],TestAcc,epoch+1)
        tb_writer.add_scalar(tags[7],TestFcross,epoch+1)
        tb_writer.add_scalar(tags[8],TestMse,epoch+1)
        tb_writer.add_scalar(tags[9],TestMae,epoch+1)
        if (epoch+1) % args.print_freq == 0:
            msg = ("epoch:{} model{} TrainLoss:{:.2f} TrainAcc:{:.2f} TrainFcross:{:.2f} TrainMse:{:.2f} TrainMae:{:.2f}"
                   "TestLoss:{:.2f} TestAcc:{:.2f} TestFcross:{:.2f} TestMse:{:.2f} TestMae:{:.2f}\n").format(
                epoch+1,args.model_names,TrainLoss,TrainAcc,TrainFcross,TrainMse,TrainMae,
                TestLoss,TestAcc,TestFcross,TestMse,TestMae
            )
            print(msg)
            f.write(msg)
            f.flush()
    msg_best = "model:{} TestMae:{:.2f}\n".format(args.model_names,min_mae)
    time_elapsed = "tranning time:{}".format(time.time() - since)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()


if __name__ == "__main__":
    if args.mode == 0:
      tb_path = "model/{}/{}".format(args.model_names,args.exp_postfix)
      tb_writer = SummaryWriter(log_dir=tb_path)
      save_path = r"/neural_network/model/reg_predict_best_model.pth"
      trainset = WeightDataset(train_dir, train_data, transform=transformer_train)
      testset = WeightDataset(test_dir, test_data, transform=transformer_train)

      train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=0, shuffle=True)
      test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=True)

      model = SeedWeightVitalityPredictor3()
      if torch.cuda.is_available():
          model = model.cuda()
      optimizer = torch.optim.Adam(
          model.parameters(),
          lr=args.lr,
          weight_decay=args.weight_decay
      )
      scheduler = CosineAnnealingLR(optimizer,T_max=args.epoch)

      train(model,optimizer,train_loader,test_loader,scheduler,tb_writer)
    elif args.mode == 1:
        model = SeedWeightVitalityPredictor3()
        model_path = "./model/{}/{}/SeedRegressionClasses/Best.pth".format(args.model_names, args.exp_postfix)
        checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model'])
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()
        results = []

        with torch.no_grad():
            for image, OriginalWeight, Day1Weight, Day2Weight, Day3Weight, Vitality,SeedId in tqdm(AllDataLoader,
                                                                                            desc="Predicting"):
                if torch.cuda.is_available():
                    image = image.cuda(non_blocking=True)
                    OriginalWeight = OriginalWeight.cuda(non_blocking=True)

                vitality_pred, weight_pred = model(image, OriginalWeight.unsqueeze(1))

                weight_pred = weight_pred.cpu().numpy()
                OriginalWeight = OriginalWeight.cpu().numpy()
                Day1Weight = Day1Weight.cpu().numpy()
                Day2Weight = Day2Weight.cpu().numpy()
                Day3Weight = Day3Weight.cpu().numpy()
                vitality_pred = torch.sigmoid(vitality_pred).cpu().numpy()
                Vitality = Vitality.cpu().numpy()
                SeedId = SeedId.cpu().numpy()

                for i in range(len(weight_pred)):
                    results.append({
                        "OriginalWeight": OriginalWeight[i],
                        "Day1WeightTrue": Day1Weight[i],
                        "Day2WeightTrue": Day2Weight[i],
                        "Day3WeightTrue": Day3Weight[i],
                        "Day1WeightPred": weight_pred[i][0],
                        "Day2WeightPred": weight_pred[i][1],
                        "Day3WeightPred": weight_pred[i][2],
                        "VitalityTrue": Vitality[i],
                        "VitalityPredProb": vitality_pred[i][0],
                        "VitalityPredLabel": int(vitality_pred[i][0] > 0.5),
                        "SeedId":SeedId[i]
                    })

        result_df = pd.DataFrame(results)
        result_path = os.path.join(exp_path, "prediction_results.csv")
        result_df.to_csv(result_path, index=False)
        print(f"result: {result_path}")
        csv = '/home/liushuai/seed/neural_network/model/resnet50/result1/prediction_results.csv'
        predict_data = pd.read_csv(csv)
        OriginalWeight = predict_data.values[:, 0]
        Day1Weight = predict_data.values[:, 4]
        Day2Weight = predict_data.values[:, 5]
        Day3Weight = predict_data.values[:, 6]
        Vitality = predict_data.values[:, 7]

        X = np.stack([OriginalWeight, Day1Weight, Day2Weight, Day3Weight], axis=1)

        scaler = StandardScaler()
        scaler_path = os.path.join(exp_path, "scaler.pkl")
        joblib.dump(scaler,scaler_path)
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Vitality, test_size=0.2, random_state=42)

        # LogisticRegression
        model_lr = LogisticRegression()
        model_lr.fit(X_train, Y_train)
        y_lr = model_lr.predict(X_test)
        BestVitalityAcc = accuracy_score(Y_test,y_lr)
        print("Logistic Regression Accuracy:", BestVitalityAcc)
        best_vitality_model = model_lr
        BestVitalityModelPath = os.path.join(exp_path, "BestVitalityModel.pkl")
        joblib.dump(best_vitality_model,BestVitalityModelPath)
        # RandomForestClassifier
        model_fc = RandomForestClassifier()
        model_fc.fit(X_train, Y_train)
        y_fc = model_fc.predict(X_test)
        FcVitalityAcc = accuracy_score(Y_test,y_fc)
        print("RandomForestClassifier Accuracy:", FcVitalityAcc)
        if FcVitalityAcc > BestVitalityAcc:
            BestVitalityAcc = FcVitalityAcc
            best_vitality_model = model_fc
            joblib.dump(best_vitality_model, BestVitalityModelPath)
        # XGBOOST
        model_xg = XGBClassifier()
        model_xg.fit(X_train, Y_train)
        y_xg = model_xg.predict(X_test)
        XGVitalityAcc = accuracy_score(Y_test, y_xg)
        print("XGVitalityAcc Accuracy:", FcVitalityAcc)
        if XGVitalityAcc > BestVitalityAcc:
            BestVitalityAcc = XGVitalityAcc
            best_vitality_model = model_xg
            joblib.dump(best_vitality_model, BestVitalityModelPath)
    else :
        model = SeedWeightVitalityPredictor3()
        model_path = "./model/{}/{}/SeedRegressionClasses/Best.pth".format(args.model_names, args.exp_postfix)
        checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model'])
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()
        results = []



