import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd  # 添加pandas支持


def load_training_metrics(folder_path):
    """加载并解析 training_metrics.json"""
    metrics_path = Path(folder_path) / "training_metrics.json"
    with open(metrics_path, 'r') as f:
        data = json.load(f)

    metrics = data['metrics']

    # 提取训练指标（跳过前两个字符串）
    train_data = {m['epoch']: m['loss'] for m in metrics['train'][2:]}  # 仅保留字典

    # 提取验证指标（跳过前5个字符串）
    val_data = {m['epoch']: m for m in metrics['val'][5:]}  # 跳过 ["epoch", "loss", ...]

    # 提取可解释性指标（仅保留字典）
    interpret_data = {m['epoch']: m for m in metrics['interpret'] if isinstance(m, dict)}

    return {
        'train': train_data,
        'val': val_data,
        'interpret': interpret_data
    }


def plot_loss_comparison(experiment_folders, output_path="loss_comparison.png"):
    """每个实验组生成独立子图"""
    num_experiments = len(experiment_folders)
    # 修改列数为3，减少行数
    cols = 3
    rows = (num_experiments + cols - 1) // cols  # 向上取整

    # 调整图表尺寸（宽度 × 高度）
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))  # 更宽更矮
    fig.suptitle("Training and Validation Loss Comparison", fontsize=16)
    sns.set(style="whitegrid")

    if num_experiments > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, folder in enumerate(experiment_folders):
        metrics = load_training_metrics(folder)
        epochs = sorted(metrics['train'].keys())
        train_losses = [metrics['train'][e] for e in epochs]
        val_losses = [metrics['val'][e]['loss'] for e in epochs]

        label = Path(folder).name
        axes[i].plot(epochs, train_losses, label=f"{label} (Train Loss)")
        axes[i].plot(epochs, val_losses, linestyle='--', label=f"{label} (Val Loss)")
        axes[i].set_title(label, fontsize=10)  # 减小标题字体
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].legend(prop={'size': 8})  # 减小图例字体

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close()


def plot_classification_accuracy(experiment_folders, output_path="accuracy_comparison.png"):
    """仅显示最后一个epoch的准确率，表格形式"""
    data = []
    for folder in experiment_folders:
        metrics = load_training_metrics(folder)
        epochs = sorted(metrics['val'].keys())
        last_epoch = epochs[-1]
        val_metrics = metrics['val'][last_epoch]

        data.append({
            'Experiment': Path(folder).name,
            'Global Acc': f"{val_metrics['global_acc']:.4f}",
            'Intermediate Acc': f"{val_metrics['intermediate_acc']:.4f}",
            'Local Acc': f"{val_metrics['local_acc']:.4f}"
        })

    df = pd.DataFrame(data)

    # 调整图表尺寸和字体
    fig, ax = plt.subplots(figsize=(10, 3 + 0.5 * len(data)))  # 增高图表
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center',
                     colWidths=[0.25, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # 减小字体以适应更多行
    table.scale(1.2, 1.2)

    plt.title("Classification Accuracy at Last Epoch", fontsize=14)
    plt.savefig(output_path)
    plt.close()


def plot_interpretability_metrics(experiment_folders, output_path="interpretability_metrics.png"):
    """仅显示最后一个epoch的可解释性指标，表格形式"""
    data = []
    for folder in experiment_folders:
        metrics = load_training_metrics(folder)
        epochs = sorted(metrics['interpret'].keys())
        last_epoch = epochs[-1]
        interpret_metrics = metrics['interpret'][last_epoch]

        data.append({
            'Experiment': Path(folder).name,
            'KL_global-mid': f"{interpret_metrics['KL_global-mid']:.4f}",
            'KL_mid-local': f"{interpret_metrics['KL_mid-local']:.4f}",
            'MI_global-mid': f"{interpret_metrics['MI_global-mid']:.4f}",
            'MI_mid-local': f"{interpret_metrics['MI_mid-local']:.4f}",
            'D-Corr_global-mid': f"{interpret_metrics.get('distance_corr_global-mid', 0):.4f}",
            'D-Corr_mid-local': f"{interpret_metrics.get('distance_corr_mid-local', 0):.4f}",
            'CCA_global-mid': f"{interpret_metrics.get('cca_similarity_global-mid', 0):.4f}",
            'CCA_mid-local': f"{interpret_metrics.get('cca_similarity_mid-local', 0):.4f}"
        })

    df = pd.DataFrame(data)
    # 定义要保存的文件路径
    file_path = f"{output_path}.txt"
    # 使用 to_csv 方法将 DataFrame 写入 TXT 文件，以 | 作为分隔符
    df.to_csv(file_path, sep='|', na_rep='nan', index=False, encoding='utf-8-sig')

    # 动态调整列宽
    col_widths = [0.15 if i == 0 else 0.1 for i in range(len(df.columns))]
    fig, ax = plt.subplots(figsize=(15, 3 + 0.5 * len(data)))  # 增宽图表
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # 更小字体适应更多列
    table.scale(1.2, 1.2)

    plt.title("Interpretability Metrics at Last Epoch", fontsize=14)
    plt.savefig(output_path)
    plt.close()


def main():
    base_dir = Path("results-0513")
    model_types = ["p2_bert", "p2_gpt2"]

    for model in model_types:
        model_dir = base_dir / model
        # 获取所有实验子文件夹
        experiment_folders = [
            str(model_dir / exp_dir) for exp_dir in os.listdir(model_dir)
            if (model_dir / exp_dir).is_dir()
        ]

        # 过滤有效实验（包含training_metrics.json的文件夹）
        valid_experiments = []
        for folder in experiment_folders:
            if Path(folder).joinpath("training_metrics.json").exists():
                valid_experiments.append(folder)

        # 按名称排序（可选）
        valid_experiments.sort()

        # 生成带模型前缀的输出文件名
        prefix = f"{model}_"
        plot_loss_comparison(
            valid_experiments,
            output_path=f"{prefix}loss_comparison.png"
        )
        plot_classification_accuracy(
            valid_experiments,
            output_path=f"{prefix}accuracy_comparison.png"
        )
        plot_interpretability_metrics(
            valid_experiments,
            output_path=f"{prefix}interpretability_metrics.png"
        )



if __name__ == "__main__":
    main()
    # dir= "/paper/script/interpret2/results/bert"
    # file_name=os.listdir(dir)
    # print(file_name)
