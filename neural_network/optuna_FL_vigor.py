#optuna  search for the best result
import os
import re
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import optuna
from fuction_file import MultiImageDataset, compute_mean_std, seed_torch
from neural_network.model.resnet50mlp import build_resnet101_18channel
from fuction_file import train_simple_3


def objective(trial, args, train_loader, val_loader, transform_list, log_file, optuna_results_dir):
    args.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    args.alpha = trial.suggest_float("alpha", 0.26, 0.38)
    args.gamma = trial.suggest_float("gamma", 1.5, 2.6)

    seed_torch(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet101_18channel(num_classes=2, pretrained=False).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    trial_log_path = os.path.join(optuna_results_dir, f"trial_{trial.number}_log.txt")

    with open(trial_log_path, 'w') as trial_log_file:
        best_para = train_simple_3(model, optimizer, train_loader, val_loader, scheduler, optuna_results_dir, args,
                                   trial_log_file)

    log_file.write(
        f"Trial {trial.number}: lr={args.lr:.6f}, alpha={args.alpha:.4f}, gamma={args.gamma:.4f}, best_para={best_para:.4f}\n")
    log_file.flush()

    return best_para


def main(args):
    image_dirs = [
        "/media/ls/办公/ls/seeddata/seeddata/prosessed_imag",
        "/media/ls/办公/ls/seeddata/seeddata/tail_image",
        "/media/ls/办公/ls/seeddata/seeddata/head_image",
        "/media/ls/办公/ls/seeddata/seeddata/destroy_image",
        "/media/ls/办公/ls/seeddata/seeddata/bw_image",
    ]
    file_path = "/home/ls/code/seed/neural_network/save_outputv8.xlsx"

    df = pd.read_excel(file_path, header=None)
    labels = df.values[:, 10]

    image_files = sorted(
        [f for f in os.listdir(image_dirs[0]) if f.lower().endswith(('.jpg', '.png', '.bmp'))],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    assert len(image_files) == len(labels), f"Image count mismatch: {len(image_files)} vs {len(labels)}"

    transform_list = []
    for dir_path in image_dirs:
        mean, std = compute_mean_std(dir_path)
        transform_list.append(transforms.Compose([
            transforms.Resize((args.re_size, args.re_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]))

    data_list = list(zip(image_files, labels))
    train_list, val_list = train_test_split(data_list, train_size=0.8, random_state=42)

    train_loader = DataLoader(MultiImageDataset(train_list, image_dirs, transform_list),
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(MultiImageDataset(val_list, image_dirs, transform_list),
                            batch_size=args.batch_size, shuffle=False, num_workers=4)

    optuna_results_dir = os.path.join("model", args.model_names, "optuna_results")
    os.makedirs(optuna_results_dir, exist_ok=True)

    total_log_path = os.path.join(optuna_results_dir, "optuna_summary.txt")
    log_file = open(total_log_path, 'w')

    study = optuna.create_study(direction="maximize")

    study.optimize(lambda trial: objective(trial, args, train_loader, val_loader, transform_list, log_file, optuna_results_dir),
                   n_trials=500, show_progress_bar=True)

    print("Number of finished trials: ", len(study.trials))
    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params_path = os.path.join(optuna_results_dir, "best_params.json")
    import json
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=4)
    print(f"Best parameters saved to {best_params_path}")

    log_file.write(f"\nBest Trial: {trial.number}\n")
    log_file.write(f"Best Value: {trial.value:.4f}\n")
    log_file.write("Best Parameters:\n")
    for key, value in trial.params.items():
        log_file.write(f"  {key}: {value}\n")

    log_file.close()
    print(f"Optuna summary saved to {total_log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, default="resnet101_18ch")
    parser.add_argument("--re_size", type=int, default=224)
    parser.add_argument("--classes_num", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--exp_postfix", type=str, default="optuna_result")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(args)