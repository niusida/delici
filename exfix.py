#python main.py --image_dir "E:\py test\austim\datasets\Image" --table_dir "E:\py test\austim\datasets\Table" --model_dir "E:\py test\austim\models" --log_dir "E:\py test\austim\logs" --use_amp
#python main.py --image_dir "E:\py test\austim\datasets\Image" --table_dir "E:\py test\austim\datasets\Table" --model_dir "E:\py test\austim\models" --log_dir "E:\py test\austim\logs" --use_amp --use_ensemble
#运行命令
#tensorboard --logdir="E:\py test\austim\logs"
#tensorboard --logdir=runs


# -*- coding: utf-8 -*-
import os
import pandas as pd

# 1. 读取CSV文件
csv_path = r"C:\Users\ASUS\Desktop\jinf.csv"  # 输入的CSV文件路径
output_path = r"C:\Users\ASUS\Desktop\jin.csv"  # 输出的CSV文件路径
folder_path = r"C:\Users\ASUS\Desktop\精分m1"

# 尝试用GBK编码读取CSV文件
try:
    df = pd.read_csv(csv_path, encoding='gbk')
except UnicodeDecodeError:
    # 如果GBK失败，可以尝试ISO-8859-1或其他编码
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# 2. 获取文件夹中的所有文件名（去掉扩展名）
file_names = set(os.path.splitext(file)[0] for file in os.listdir(folder_path))

# 3. 过滤掉第一列(label列)中没有匹配文件名的行
# 假设第一列的列名为 'label'，根据你的CSV文件实际情况进行修改
df_filtered = df[df['label'].astype(str).apply(lambda x: x in file_names)]

# 4. 保存过滤后的CSV文件
df_filtered.to_csv(output_path, index=False, encoding='utf-8')

print(f"过滤完成，结果已保存到: {output_path}")






# -*- coding: utf-8 -*-


# main.py

# -*- coding: utf-8 -*-

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, AlzheimerDataset
from train import train_one_epoch, evaluate
from model import AlzheimerCNN
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    set_seed(args.seed)  # Set random seed

    # Data directories
    image_dir = args.image_dir  # Image data directory
    table_dir = args.table_dir  # Table data directory

    train_excel = os.path.join(table_dir, 'train.xlsx')

    # Load all train data
    filepaths, labels, tabular_features = load_data(image_dir, train_excel, split='train')

    print("Total samples:", len(filepaths))

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    class_names = [str(cn) for cn in label_encoder.classes_]  # Ensure class names are strings
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Save label encoder
    encoder_scaler_dir = args.model_dir
    os.makedirs(encoder_scaler_dir, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(encoder_scaler_dir, 'label_encoder.joblib'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize K-Fold
    k_folds = args.k_folds
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)

    # Convert tabular_features to numpy array for indexing
    tabular_features = np.array(tabular_features)

    # Fit a global scaler on the entire dataset
    scaler = StandardScaler()
    tabular_features = scaler.fit_transform(tabular_features)
    joblib.dump(scaler, os.path.join(encoder_scaler_dir, 'tabular_scaler.joblib'))

    # Initialize lists to collect overall predictions and true labels
    overall_preds = []
    overall_true = []

    # To store loss per epoch across all folds
    all_fold_train_losses = []
    all_fold_val_losses = []

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(filepaths)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        # Sample elements according to the split indices
        train_filepaths = [filepaths[i] for i in train_ids]
        train_labels = [labels_encoded[i] for i in train_ids]
        train_tabular = tabular_features[train_ids]

        val_filepaths = [filepaths[i] for i in val_ids]
        val_labels = [labels_encoded[i] for i in val_ids]
        val_tabular = tabular_features[val_ids]

        # Create Datasets
        train_dataset = AlzheimerDataset(
            train_filepaths,
            train_labels,
            train_tabular,
            transform=transform_train,
            label_encoder=label_encoder
        )
        val_dataset = AlzheimerDataset(
            val_filepaths,
            val_labels,
            val_tabular,
            transform=transform_val,
            label_encoder=label_encoder
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

        # Initialize model
        model = AlzheimerCNN(num_classes=num_classes, tabular_input_size=train_dataset.tabular_features.shape[1]).to(device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        # Early Stopping parameters
        patience = args.patience
        trigger_times = 0
        best_val_loss = float('inf')

        # Train the model for a fixed number of epochs
        epochs = args.epochs
        fold_train_losses = []
        fold_val_losses = []
        best_model_path = os.path.join(args.model_dir, f'best_model_fold_{fold+1}.pth')

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.use_amp)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            scheduler.step(val_loss)

            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)

            # Log to TensorBoard
            writer.add_scalars(f'Fold_{fold+1}/Loss', {'train': train_loss, 'val': val_loss}, epoch+1)
            writer.add_scalars(f'Fold_{fold+1}/Accuracy', {'train': train_acc, 'val': val_acc}, epoch+1)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                # Save the best model
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model for fold {fold+1} saved with Val Loss: {best_val_loss:.4f}")
            else:
                trigger_times += 1
                print(f"No improvement for fold {fold+1}. Trigger times: {trigger_times}/{patience}")
                if trigger_times >= patience:
                    print("Early stopping!")
                    break

        # Load the best model for evaluation
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # Evaluate on validation set
        val_loader_eval = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
        with torch.no_grad():
            for inputs, tabular, labels in val_loader_eval:
                inputs, tabular, labels = inputs.to(device), tabular.to(device), labels.to(device)
                outputs = model(inputs, tabular)
                _, predicted = torch.max(outputs.data, 1)
                overall_preds.append(predicted.item())
                overall_true.append(labels.item())

        # Append fold losses for plotting
        all_fold_train_losses.append(fold_train_losses)
        all_fold_val_losses.append(fold_val_losses)

    # Compute overall accuracy
    overall_acc = accuracy_score(overall_true, overall_preds)
    print(f"\n=== K-Fold Cross-Validation Results ===")
    print(f"Overall Accuracy: {overall_acc:.4f}")

    # Classification Report
    report = classification_report(overall_true, overall_preds, target_names=class_names)
    print("\nClassification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(overall_true, overall_preds)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Optionally, save the overall predictions and true labels to CSV
    results_df = pd.DataFrame({
        'true_label': overall_true,
        'predicted_label': overall_preds
    })
    results_df.to_csv('kfold_results.csv', index=False)
    print("K-Fold results saved to 'kfold_results.csv'.")

    writer.close()

def collate_fn(batch):
    """Custom collate function to handle potential None samples."""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)






