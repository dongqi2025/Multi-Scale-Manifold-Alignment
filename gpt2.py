import hashlib
import json
from collections import defaultdict
from datetime import datetime

import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import GPT2Model, GPT2Tokenizer  # 替换Bert相关导入
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mutual_info_score
from pathlib import Path
from interpret_eval_gpu import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
semantic_scales = {
    "local": [0, 1, 2, 3],  # 前4层（0-based索引）
    "intermediate": [4, 5, 6, 7],  # 中间层
    "global": [8, 9, 10, 11]  # 后4层
}


# 自定义数据集类
class MSMADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.texts = data["texts"]
        self.global_labels = data["global_labels"]
        self.intermediate_labels = data["intermediate_labels"]
        self.local_labels = data["local_labels"]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'global_label': torch.tensor(self.global_labels[idx], dtype=torch.long).squeeze(),
            'intermediate_label': torch.tensor(self.intermediate_labels[idx], dtype=torch.long).squeeze(),
            'local_label': torch.tensor(self.local_labels[idx], dtype=torch.long).squeeze()
        }


# 多尺度BERT特征提取器
class MultiscaleGPT2(nn.Module):
    def __init__(self, model_path='/ossfs/workspace/model/openai-community__gpt2'):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            model_path,
            output_hidden_states=True
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # 获取所有隐藏层

        # 保持原有池化逻辑
        def pool_layers(layer_indices):
            pooled = []
            for i in layer_indices:
                layer_pool = torch.mean(hidden_states[i], dim=1)
                pooled.append(layer_pool)
            return torch.mean(torch.stack(pooled), dim=0)

        h_local = pool_layers(semantic_scales["local"])
        h_intermediate = pool_layers(semantic_scales["intermediate"])
        h_global = pool_layers(semantic_scales["global"])

        return h_local, h_intermediate, h_global


# MINE互信息估计器
class Mine(nn.Module):
    def __init__(self, input_dim=768 * 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, x, y):
        joint = torch.cat((x, y), dim=1)
        return self.net(joint)


# Procrustes对齐模块
def procrustes_align(source, target):
    # 中心化
    src_centered = source - torch.mean(source, dim=0)
    tgt_centered = target - torch.mean(target, dim=0)

    # SVD分解
    U, S, Vt = torch.linalg.svd(torch.mm(src_centered.T, tgt_centered))
    rotation = torch.mm(U, Vt)

    # 应用旋转
    aligned = torch.mm(src_centered, rotation)
    return aligned


# 新增训练结果保存类
class TrainingRecorder:
    def __init__(self):
        self.metrics = {
            'train': ['epoch', 'loss'],
            'val': ['epoch', 'loss', 'global_acc', 'intermediate_acc', 'local_acc'],
            'interpret': []
        }
        # 新增元数据记录
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'metric_schema': self._init_schema()
        }

    def _init_schema(self):
        """定义指标结构模式用于验证"""
        return {
            'train': ['epoch', 'loss'],
            'val': ['epoch', 'loss'],
            'interpret': {
                'required': ['epoch', 'KL_global-mid', 'KL_mid-local',
                             'MI_global-mid', 'MI_mid-local'],
                'optional': ['distance_corr_global-mid', 'distance_corr_mid-local',
                             'cca_similarity_global-mid', 'cca_similarity_mid-local',
                             'attention_entropy_layer_*', 'global_attention_var']
            }
        }

    def add_metrics(self, mode, metrics_dict):
        """增强型指标添加方法，支持动态结构验证"""
        # 结构验证
        if mode not in self.metrics:
            raise ValueError(f"Invalid mode: {mode}. Available modes: {list(self.metrics.keys())}")

        if mode == 'interpret':
            self._validate_interpret_metrics(metrics_dict)

        self.metrics[mode].append(metrics_dict)

    def _validate_interpret_metrics(self, metrics):
        """可解释性指标结构验证"""
        required = self.metadata['metric_schema']['interpret']['required']
        for key in required:
            if key not in metrics:
                raise ValueError(f"Missing required interpret metric: {key}")

    def save_to_json(self, filename):
        """增强版保存方法，包含元数据"""
        full_data = {
            'metadata': self.metadata,
            'metrics': self.metrics
        }
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)  # 新增目录创建

        with open(filename, 'w') as f:
            json.dump(full_data, f, indent=2, cls=DateTimeEncoder)

    def save_interpret_report(self, filename):
        """生成增强版可视化报告"""
        report = {
            'semantic_metrics': {
                'KL_divergence': {
                    'global-mid': self._extract_trend('KL_global-mid'),
                    'mid-local': self._extract_trend('KL_mid-local')
                },
                'Mutual_Information': {
                    'global-mid': self._extract_trend('MI_global-mid'),
                    'mid-local': self._extract_trend('MI_mid-local')
                }
            },
            'correlation_analysis': {
                'Distance_Correlation': self._extract_correlation('distance_corr'),
                'CCA_Similarity': self._extract_correlation('cca_similarity')
            },
            'attention_analysis': self._analyze_attention_patterns(),
            'classification_performance': {
                'global_accuracy': self._extract_trend('global_acc'),
                'intermediate_accuracy': self._extract_trend('intermediate_acc'),
                'local_accuracy': self._extract_trend('local_acc')
            }
        }
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

    def _extract_trend(self, metric_key):
        """提取指定指标的演化趋势"""
        return [m.get(metric_key, None) for m in self.metrics['interpret']]

    def _extract_correlation(self, prefix):
        """提取相关分析指标"""
        return {
            'global-mid': self._extract_trend(f"{prefix}_global-mid"),
            'mid-local': self._extract_trend(f"{prefix}_mid-local")
        }

    def _analyze_attention_patterns(self):
        """分析注意力模式趋势"""
        patterns = {
            'layer_entropy': defaultdict(list),
            'global_variance': []
        }

        for entry in self.metrics['interpret']:
            # 收集各层注意力熵
            for key in entry:
                if key.startswith('attention_entropy_layer_'):
                    layer_num = key.split('_')[-1]
                    patterns['layer_entropy'][f"layer_{layer_num}"].append(entry[key])

            # 收集全局注意力方差
            if 'attention_dynamics_global_attention_var' in entry:
                patterns['global_variance'].append(entry['attention_dynamics_global_attention_var'])

        return patterns


class AlignmentVisualizer:
    def __init__(self, n_samples=500, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.palette = sns.color_palette("husl", 3)  # 使用三种颜色对应三个层级
        self.tsne_params = {
            'perplexity': 30,
            'n_iter': 1000,
            'metric': 'cosine'
        }

    def _safe_sample(self, data, labels, group_name):
        """增强鲁棒性的采样方法"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        sampled_indices = []

        for lbl, count in zip(unique_labels, counts):
            indices = np.where(labels == lbl)[0]
            sample_size = max(1, int(self.n_samples * (count / len(labels))))

            # 自动调整采样策略
            try:
                sampled = np.random.choice(
                    indices,
                    size=min(sample_size, len(indices)),
                    replace=False
                )
                sampled_indices.extend(sampled)
            except ValueError as e:
                print(f"采样失败（{group_name} - 类别{lbl}）: {str(e)}")
                continue
        return np.array(sampled_indices)

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

    def _reduce_dim(self, embeddings):
        return TSNE(**self.tsne_params).fit_transform(embeddings)

    def plot_alignment(self, embeddings_dict, labels, save_path):
        plt.figure(figsize=(24, 18))

        for idx, scale in enumerate(['global', 'intermediate', 'local']):
            ax = plt.subplot(1, 3, idx + 1)
            embeddings = embeddings_dict[scale]

            # 动态调整采样量
            min_samples = min([emb.shape[0] for emb in embeddings.values()])
            sample_size = min(self.n_samples, min_samples)  # 保证各组采样量一致

            # 绘制每个实验组
            for group_idx, (group_name, emb) in enumerate(embeddings.items()):
                # 分层采样保证类别平衡
                sampled_indices = []
                unique_labels = np.unique(labels)

                for lbl in unique_labels:
                    indices = np.where(labels == lbl)[0]
                    size_per_class = max(1, int(sample_size / len(unique_labels)))

                    # 允许少量重复采样
                    sampled = np.random.choice(
                        indices,
                        size=size_per_class,
                        replace=len(indices) < size_per_class
                    )
                    sampled_indices.extend(sampled)

                # 降维可视化
                reduced = self._reduce_dim(emb[sampled_indices])
                ax.scatter(
                    reduced[:, 0], reduced[:, 1],
                    color=self.palette[group_idx],
                    label=group_name,
                    alpha=0.7,
                    edgecolor='w',
                    s=60,
                    linewidth=0.5
                )

            ax.set_title(f"{scale.capitalize()} Layer", fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True, alpha=0.3)

        # 统一图例（显示所有组）
        handles, labels = ax.get_legend_handles_labels()
        plt.figlegend(
            handles, labels,
            loc='lower center',
            ncol=4,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.05)
        )

        plt.tight_layout(pad=3.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class DateTimeEncoder(json.JSONEncoder):
    """支持datetime对象的JSON编码"""

    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


# 数据加载分离
def get_data_loaders(config):
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_path'])
    tokenizer.pad_token = tokenizer.eos_token  # GPT2需要手动设置pad token
    with open(config['data_path'], 'r') as f:
        data = json.load(f)

    # 创建完整数据集对象
    full_dataset = MSMADataset(data, tokenizer)

    # 基于实际样本数划分
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    # 直接划分数据集对象
    train_data, val_data = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )

    # 创建DataLoader时直接使用划分后的Subset
    return (
        DataLoader(train_data, batch_size=config['batch_size'], shuffle=True),
        DataLoader(val_data, batch_size=config['batch_size'])
    )


def compute_kl_divergence(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """改进的高斯KL散度计算，增强数值稳定性"""
    eps = 1e-6  # 增大平滑项
    batch_size, dim = h1.shape

    # 使用样本方差（除以n），避免小批次下的不稳定
    mu1 = torch.mean(h1, dim=0)
    sigma1 = torch.var(h1, dim=0, unbiased=False) + eps  # 关键修改：unbiased=False
    mu2 = torch.mean(h2, dim=0)
    sigma2 = torch.var(h2, dim=0, unbiased=False) + eps

    # 确保方差非负（理论上已保证，防止数值误差）
    sigma1 = torch.clamp(sigma1, min=eps)
    sigma2 = torch.clamp(sigma2, min=eps)

    # 计算KL散度
    kl = 0.5 * torch.sum(sigma1 / sigma2)
    kl += 0.5 * torch.sum((mu1 - mu2) ** 2 / sigma2)
    kl += 0.5 * torch.sum(torch.log(sigma2) - torch.log(sigma1))
    kl -= 0.5 * dim  # 修正项

    return kl


def estimate_mutual_info(h1: torch.Tensor, h2: torch.Tensor, num_bins=20) -> float:
    """改进的互信息估计，使用自适应分箱"""
    # 转换为numpy并标准化
    h1_np = (h1.cpu().numpy() - np.mean(h1.cpu().numpy())) / np.std(h1.cpu().numpy())
    h2_np = (h2.cpu().numpy() - np.mean(h2.cpu().numpy())) / np.std(h2.cpu().numpy())

    # 自适应分箱
    def adaptive_binning(data):
        q = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.unique(np.percentile(data, q))
        return np.digitize(data, bin_edges) - 1  # 从0开始编号

    # 离散化
    disc_h1 = adaptive_binning(h1_np.flatten())
    disc_h2 = adaptive_binning(h2_np.flatten())

    return mutual_info_score(disc_h1, disc_h2)


# 完整MSMA模型
class MSMA(nn.Module):
    def __init__(self, config):
        # 初始化所有组件
        super().__init__()
        self.config = config
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"self.device :{self.device}")

        # 模型组件
        self.model = MultiscaleGPT2(config['model_path']).to(self.device)

        self.global_classifier = nn.Linear(768, config['num_global_classes']).to(self.device)
        self.intermediate_classifier = nn.Linear(768, config['num_intermediate_classes']).to(self.device)
        self.local_classifier = nn.Linear(768, config['num_local_classes']).to(self.device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config['learning_rate'])
        self.recorder = TrainingRecorder()
        self.train_loader, self.val_loader = get_data_loaders(config)

        # 信息对齐组件
        self.geometric_alignment = config.get('geometric_alignment', False)
        self.info_alignment = config.get('info_alignment', False)
        self.curvature_reg = config.get('curvature_reg', False)

        # 初始化信息对齐组件
        if self.info_alignment:
            self.mine_global_mid = Mine()
            self.mine_mid_local = Mine()

        self.to(self.device)

    def assess_interpretability(self, dataloader, mode='val'):
        """优化后的可解释性评估方法"""
        self.eval()
        # 初始化累计器
        stats = defaultdict(list)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # 随机采样20%数据
            sampled_loader = self._get_sampled_loader(dataloader, sample_ratio=0.5)

            # 使用内存缓存加速
            cache = []
            for batch in sampled_loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                h_local, h_mid, h_global = self.model(batch['input_ids'], batch['attention_mask'])

                # 使用半精度并缓存特征
                cache.append((
                    h_global.half().cpu(),
                    h_mid.half().cpu(),
                    h_local.half().cpu()
                ))

            # 合并特征（GPU加速）
            h_global = torch.cat([x[0] for x in cache]).float().to(self.device)
            h_mid = torch.cat([x[1] for x in cache]).float().to(self.device)
            h_local = torch.cat([x[2] for x in cache]).float().to(self.device)

            # 并行计算指标
            results = self._parallel_compute_metrics(h_global, h_mid, h_local)

        return results

    def _parallel_compute_metrics(self, h_global, h_mid, h_local):
        """并行化指标计算"""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as executor:
            # 提交计算任务
            future_kl = executor.submit(self._compute_kl_async, h_global, h_mid, h_local)
            future_mi = executor.submit(self._compute_mi_async, h_global, h_mid, h_local)
            future_dcorr = executor.submit(self._compute_dcorr_async, h_global, h_mid, h_local)
            future_cca = executor.submit(self._compute_cca_async, h_global, h_mid, h_local)

            # 等待结果
            kl_gm, kl_ml = future_kl.result()
            mi_gm, mi_ml = future_mi.result()
            dcorr_gm, dcorr_ml = future_dcorr.result()
            cca_gm, cca_ml = future_cca.result()

        return {
            'KL_global-mid': kl_gm,
            'KL_mid-local': kl_ml,
            'MI_global-mid': mi_gm,
            'MI_mid-local': mi_ml,
            'distance_corr_global-mid': dcorr_gm,
            'distance_corr_mid-local': dcorr_ml,
            'cca_similarity_global-mid': cca_gm,
            'cca_similarity_mid-local': cca_ml
        }

    def _compute_kl_async(self, h_global, h_mid, h_local):
        """优化后的KL散度计算"""
        return (
            compute_kl_divergence(h_global, h_mid).item(),
            compute_kl_divergence(h_mid, h_local).item()
        )

    def _compute_mi_async(self, h_global, h_mid, h_local):
        """基于GPU的快速互信息估计"""
        return (
            fast_mi_estimate(h_global, h_mid),
            fast_mi_estimate(h_mid, h_local)
        )

    def _compute_dcorr_async(self, h_global, h_mid, h_local):
        """批量距离相关性计算"""
        return (
            batch_distance_corr(h_global, h_mid),
            batch_distance_corr(h_mid, h_local)
        )

    def _compute_cca_async(self, h_global, h_mid, h_local):
        """优化后的CCA计算"""
        return (
            fast_cca(h_global, h_mid, pca_dim=32),
            fast_cca(h_mid, h_local, pca_dim=32)
        )

    def _get_sampled_loader(self, dataloader, sample_ratio=0.2):
        """生成采样后的DataLoader"""
        original_size = len(dataloader.dataset)
        sample_size = int(original_size * sample_ratio)
        indices = torch.randperm(original_size)[:sample_size]
        return DataLoader(Subset(dataloader.dataset, indices),
                          batch_size=dataloader.batch_size,
                          shuffle=False)

    def forward(self, input_ids, attention_mask):
        h_local, h_intermediate, h_global = self.model(input_ids, attention_mask)

        # 几何对齐
        if self.geometric_alignment:
            h_global = procrustes_align(h_global, h_intermediate)
            h_intermediate = procrustes_align(h_intermediate, h_local)

        # 分类预测
        global_logits = self.global_classifier(h_global)
        intermediate_logits = self.intermediate_classifier(h_intermediate)
        local_logits = self.local_classifier(h_local)

        return global_logits, intermediate_logits, local_logits

    def compute_loss(self, batch, epoch):

        # 前向传播
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        global_labels = batch['global_label'].to(self.device)
        intermediate_labels = batch['intermediate_label'].to(self.device)
        local_labels = batch['local_label'].to(self.device)

        # 获取各尺度表示
        h_local, h_intermediate, h_global = self.model(input_ids, attention_mask)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_dict = {}

        # 分类损失
        global_loss = nn.CrossEntropyLoss()(self.global_classifier(h_global), global_labels)
        intermediate_loss = nn.CrossEntropyLoss()(self.intermediate_classifier(h_intermediate), intermediate_labels)
        local_loss = nn.CrossEntropyLoss()(self.local_classifier(h_local), local_labels)
        cls_loss = (global_loss + intermediate_loss + local_loss) / 3

        total_loss = total_loss + cls_loss

        loss_dict['cls_loss'] = cls_loss.item()

        # 几何对齐损失
        if self.geometric_alignment:
            # 最小二乘对齐损失
            ls_loss = torch.mean(torch.norm(h_global - h_intermediate, p=2, dim=1)) + \
                      torch.mean(torch.norm(h_intermediate - h_local, p=2, dim=1))
            total_loss = total_loss + self.config['lambda_geo'] * ls_loss
            loss_dict['geo_loss'] = ls_loss.item()

        # 信息对齐损失
        if self.info_alignment:
            # MINE互信息估计
            mi_gm = torch.mean(self.mine_global_mid(h_global, h_intermediate)) - \
                    torch.log(torch.mean(torch.exp(self.mine_global_mid(h_global, h_intermediate.detach()))))
            mi_ml = torch.mean(self.mine_mid_local(h_intermediate, h_local)) - \
                    torch.log(torch.mean(torch.exp(self.mine_mid_local(h_intermediate, h_local.detach()))))
            mi_loss = -(mi_gm + mi_ml)
            total_loss = total_loss + self.config['lambda_info'] * mi_loss
            loss_dict['info_loss'] = mi_loss.item()

        # 曲率正则化
        if self.curvature_reg:
            # 简化实现：参数二阶范数约束
            params = torch.cat([p.view(-1) for p in self.parameters()])
            curv_reg = torch.norm(params, p=2)
            total_loss = total_loss + self.config['lambda_curv'] * curv_reg
            loss_dict['curv_loss'] = curv_reg.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

    def train_epoch(self, epoch):
        self.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # 数据转移到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 计算损失
            loss, loss_dict = self.compute_loss(batch, epoch)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()

            # 记录指标
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

        return epoch_loss / len(self.train_loader)

    def validate(self, epoch):
        self.eval()
        val_loss = 0.0
        global_correct = 0
        intermediate_correct = 0
        local_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                global_logits, intermediate_logits, local_logits = self(
                    batch['input_ids'],
                    batch['attention_mask']
                )

                # 计算损失
                loss, _ = self.compute_loss(batch, epoch)
                val_loss += loss.item()

                # 计算准确率
                global_correct += (torch.argmax(global_logits, 1) == batch['global_label']).sum().item()
                intermediate_correct += (
                        torch.argmax(intermediate_logits, 1) == batch['intermediate_label']).sum().item()
                local_correct += (torch.argmax(local_logits, 1) == batch['local_label']).sum().item()
                total_samples += batch['global_label'].size(0)

            avg_val_loss = val_loss / len(self.val_loader)
            return {
                "loss": avg_val_loss,
                "global_acc": global_correct / total_samples,
                "intermediate_acc": intermediate_correct / total_samples,
                "local_acc": local_correct / total_samples
            }

    def compute_accuracy(self, logits, labels):
        """通用准确率计算方法"""
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        return correct / labels.size(0)

    def run(self):
        self.best_val_loss = float('inf')
        # 新增：用于跟踪特征演化
        epoch_features_cache = []
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 0

        for epoch in range(start_epoch, self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            val_loss = val_metrics["loss"]

            self.recorder.add_metrics('train', {'epoch': epoch, 'loss': train_loss})
            self.recorder.add_metrics('val', {
                'epoch': epoch,
                'loss': val_loss,
                'global_acc': val_metrics["global_acc"],
                'intermediate_acc': val_metrics["intermediate_acc"],
                'local_acc': val_metrics["local_acc"]
            })

            print(f"Epoch {epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"Global Acc: {val_metrics['global_acc'] * 100:.2f}% | "
                  f"Intermediate Acc: {val_metrics['intermediate_acc'] * 100:.2f}% | "
                  f"Local Acc: {val_metrics['local_acc'] * 100:.2f}%")

            # === 新增模型保存逻辑 ===
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                # 清理旧的最佳模型
                for old_best in save_dir.glob("best_model_epoch*.pt"):
                    old_best.unlink()

                self.best_val_loss = val_loss
                # best_path = save_dir / f"best_model_epoch{epoch}.pt"
                # self.save_checkpoint(epoch=epoch, path=best_path)
                # print(f"New best model saved: {best_path}")

            if epoch % 3 == 0:
                print(f"start assess_interpretability ... epoch:{epoch}")

                interpret_metrics = self.assess_interpretability(self.val_loader)
                # 新增：特征演化分析
                if len(epoch_features_cache) > 1:
                    evolution_metrics = track_feature_evolution(epoch_features_cache[-3:])
                    interpret_metrics.update(evolution_metrics)
                # 将嵌套字典展开为平面结构以适应JSON存储
                flat_metrics = {}
                for key, value in interpret_metrics.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flat_metrics[f"{key}_{sub_key}"] = sub_value
                    else:
                        flat_metrics[key] = value

                self.recorder.add_metrics('interpret', {
                    'epoch': epoch,
                    **flat_metrics
                })
                checkpoint_path = save_dir / "latest_checkpoint.pt"
                # self.save_checkpoint(epoch=epoch, path=checkpoint_path)

        final_path = save_dir / "final_model.pt"
        # self.save_checkpoint(epoch=self.config['epochs'], path=final_path)

        # 最终保存时生成诊断报告
        import os
        # os.system("cd /ossfs/workspace/checkpoints_20k/ && du -h -d 2")
        self.recorder.save_to_json(f"{self.config['save_dir']}/training_metrics.json")
        self.recorder.save_interpret_report(f"{self.config['save_dir']}/interpret_report.json")

    def save_checkpoint(self, epoch: int, path: Path):
        """增强鲁棒性的保存方法"""
        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        # 保存内容增加哈希校验
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'metrics': self.recorder.metrics,
            'checksum': hashlib.md5(str(self.state_dict()).encode()).hexdigest()  # 简易校验
        }

        torch.save(checkpoint, str(path))

    def get_embeddings(self, dataloader):
        """获取所有样本的多尺度嵌入表示"""
        self.eval()
        embeddings = {
            'global': [],
            'intermediate': [],
            'local': []
        }
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                h_local, h_mid, h_global = self.model(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )

                embeddings['global'].append(h_global.cpu())
                embeddings['intermediate'].append(h_mid.cpu())
                embeddings['local'].append(h_local.cpu())
                labels.append(inputs['global_label'].cpu())
        return {
            'global': torch.cat(embeddings['global']),
            'intermediate': torch.cat(embeddings['intermediate']),
            'local': torch.cat(embeddings['local']),
            'labels': torch.cat(labels)
        }


def save_embeddings(embeddings, save_dir):
    """保存嵌入数据到指定目录"""
    emb_dir = Path(save_dir) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    np.save(emb_dir / "global.npy", embeddings['global'])
    np.save(emb_dir / "intermediate.npy", embeddings['intermediate'])
    np.save(emb_dir / "local.npy", embeddings['local'])
    np.save(emb_dir / "labels.npy", embeddings['labels'])


def load_embeddings(save_dir):
    """从指定目录加载嵌入数据"""
    emb_dir = Path(save_dir) / "embeddings"
    return {
        'global': np.load(emb_dir / "global.npy"),
        'intermediate': np.load(emb_dir / "intermediate.npy"),
        'local': np.load(emb_dir / "local.npy"),
        'labels': np.load(emb_dir / "labels.npy")
    }


def embeddings_exist(save_dir):
    """检查嵌入数据是否已存在"""
    emb_dir = Path(save_dir) / "embeddings"
    required_files = {'global.npy', 'intermediate.npy', 'local.npy', 'labels.npy'}
    existing_files = {f.name for f in emb_dir.glob('*') if f.is_file()}
    return required_files.issubset(existing_files)

# 在路径创建时添加校验
def safe_create_dir(path):
    """安全创建目录并返回Path对象"""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except PermissionError:
        print(f"权限不足无法创建目录: {path}")
        raise
    except Exception as e:
        print(f"创建目录失败: {str(e)}")
        raise


if __name__ == "__main__":

    experiment_groups = {
        "full_msma": {
            "geometric_alignment": True,
            "info_alignment": True,
            "curvature_reg": True,
            "lambda_geo": 0.1,
            "lambda_info": 0.1,
            "lambda_curv": 0.01,
            "save_dir": "./p2_gpt2/full_msma"
        },
        "no_geo": {
            "geometric_alignment": False,
            "lambda_geo": 0,  # 完全关闭几何对齐
            "lambda_info": 0.1,
            "lambda_curv": 0.01,
            "info_alignment": True,
            "curvature_reg": True,
            "save_dir": "./p2_gpt2/no_geo"
        },
        "no_info": {
            "info_alignment": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,  # 完全关闭信息对齐
            "lambda_curv": 0.01,
            "geometric_alignment": True,
            "curvature_reg": True,
            "save_dir": "./p2_gpt2/no_info"
        },
        "no_curv": {
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0.1,
            "lambda_curv": 0,  # 完全关闭曲率正则
            "geometric_alignment": True,
            "info_alignment": True,
            "save_dir": "./p2_gpt2/no_curv"
        },
        "baseline": {
            "geometric_alignment": False,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/baseline"
        },
        "only_geo_0.1": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_0.1"
        },
        "only_info": {
            "geometric_alignment": False,
            "info_alignment": True,
            "curvature_reg": False,
            "lambda_geo": 0,
            "lambda_info": 0.1,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_info"
        },
        "only_curv": {
            "geometric_alignment": False,
            "info_alignment": False,
            "curvature_reg": True,
            "lambda_geo": 0,
            "lambda_info": 0,
            "lambda_curv": 0.01,
            "save_dir": "./p2_gpt2/only_curv"
        },
        "only_geo_0.2": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.2,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_0.2"
        },
        "only_geo_0.3": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_0.3"
        },
        "only_geo_0.4": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_0.4"
        },
        "only_geo_0.5": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_0.5"
        },
        "only_geo_0.6": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_0.6"
        },
        "only_geo_0.7": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_0.7"
        },
        "only_geo_0.8": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_0.8"
        },
        "only_geo_0.9": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_0.9"
        },
        "only_geo_1": {
            "geometric_alignment": True,
            "info_alignment": False,
            "curvature_reg": False,
            "lambda_geo": 0.1,
            "lambda_info": 0,
            "lambda_curv": 0,
            "save_dir": "./p2_gpt2/only_geo_1"
        },
    }
    base_config = {
        'model_path': '/ossfs/workspace/model/openai-community__gpt2',
        'data_path': '/ossfs/workspace/p2/data/preprocessed_data_20k.txt',
        'learning_rate': 1e-5,
        'batch_size': 128,
        'epochs': 15,
        'num_global_classes': 62,
        'num_intermediate_classes': 3,
        'num_local_classes': 3,
        'save_dir': './results'  # 新增基础保存路径
    }
    # 初始化存储结构
    all_embeddings = {
        'global': {},
        'intermediate': {},
        'local': {}
    }

    labels_cache = None
    base_save_dir = Path(base_config.get('save_dir', './results'))

    for group_name, group_config in experiment_groups.items():
        final_save_path = Path(base_config['save_dir']) / group_config['save_dir']

        final_config = {
            **base_config,
            **group_config,
            'save_dir': str(final_save_path)
        }

        # 创建保存目录
        save_dir = safe_create_dir(final_config['save_dir'])

        # 检查是否已有训练结果
        if embeddings_exist(save_dir):
            print(f"\n=== Loading cached data for {group_name} ===")
            embeddings = load_embeddings(save_dir)
        else:
            print(f"\n=== Training {group_name} ===")
            model = MSMA(final_config)
            model.run()

            # 保存配置
            with open(save_dir / "config.json", 'w') as f:
                json.dump(final_config, f, indent=2)

            # 获取并保存嵌入
            val_loader = model.val_loader
            embeddings = model.get_embeddings(val_loader)
            save_embeddings({
                'global': embeddings['global'].numpy(),
                'intermediate': embeddings['intermediate'].numpy(),
                'local': embeddings['local'].numpy(),
                'labels': embeddings['labels'].numpy()
            }, save_dir)

        # 更新全局存储
        all_embeddings['global'][group_name] = embeddings['global']
        all_embeddings['intermediate'][group_name] = embeddings['intermediate']
        all_embeddings['local'][group_name] = embeddings['local']
        labels_cache = embeddings['labels']
        print("\n=== Generating Alignment Visualization ===")
        vis = AlignmentVisualizer(n_samples=1000)

        group_embeddings = {
            'global': all_embeddings['global'][group_name],
            'intermediate': all_embeddings['intermediate'][group_name],
            'local': all_embeddings['local'][group_name]
        }
        vis.plot_single_group(
            group_embeddings,
            labels_cache,
            group_name,
            save_dir  # 直接使用已创建的路径
        )

