import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

class Visualizer:
    def __init__(self, tsne_params):
        self.tsne_params = tsne_params

    """修改后的可视化函数（顺序调整为Local-Intermediate-Global）"""
    def plot_single_group(self, embeddings, labels, group_name, save_dir):
        """修改后的可视化函数（顺序调整为Local-Intermediate-Global）"""
        plt.figure(figsize=(18, 6))
        save_path = Path(save_dir) / "alignment.png"

        # 强制指定顺序
        ordered_scales = ['local', 'intermediate', 'global']

        for idx, scale in enumerate(ordered_scales):
            ax = plt.subplot(1, 3, idx + 1)
            emb = embeddings[scale]  # 按指定顺序获取数据

            # 安全采样
            sampled_indices = self._safe_sample(emb, labels, f"{group_name}-{scale}")
            if len(sampled_indices) == 0:
                print(f"跳过 {group_name}-{scale}，无有效样本")
                continue

            try:
                reduced = TSNE(**self.tsne_params).fit_transform(emb[sampled_indices])
            except Exception as e:
                print(f"降维失败 {group_name}-{scale}: {str(e)}")
                continue

            # 绘制散点图（新增颜色条）
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1],
                c=labels[sampled_indices],
                cmap='viridis',
                alpha=0.7,
                edgecolor='w',
                s=40,
                vmin=min(labels),  # 新增颜色范围控制
                vmax=max(labels)
            )

            # 添加颜色条到最右侧子图
            if idx == 2:  # 最后一个子图（Global）添加颜色条
                plt.colorbar(scatter, ax=ax, label='Class', pad=0.02)

            ax.set_title(f"{scale.capitalize()} Layer", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f"Feature Alignment - {group_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"可视化已保存至: {save_path}")

    def _safe_sample(self, emb, labels, name):
        # 实现安全采样逻辑
        pass