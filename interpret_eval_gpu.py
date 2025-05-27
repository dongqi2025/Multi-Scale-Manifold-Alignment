import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from torch import cosine_similarity

def feature_importance_analysis(model, sample):
    explainer = shap.DeepExplainer(model, sample)
    shap_values = explainer.shap_values(sample)

    importance_metrics = {
        'global_feat_imp': np.mean(np.abs(shap_values[0])),
        "local_feat_imp": np.mean(np.abs((shap_values[2])))
    }
    return importance_metrics


def compute_distance_corr(h1: torch.Tensor, h2: torch.Tensor, pca_dim=10) -> float:
    """添加PCA降维后计算距离相关系数"""
    h1_np = h1.cpu().detach().numpy()
    h2_np = h2.cpu().detach().numpy()
    return dcor.distance_correlation(h1_np, h2_np)


from sklearn.exceptions import ConvergenceWarning
import warnings


def cca_similarity(h1: torch.Tensor, h2: torch.Tensor) -> float:
    """鲁棒的CCA相似度计算 (CUDA兼容版)"""
    # 确保数据在CPU并转换格式
    h1_np = h1.cpu().detach().numpy().astype(np.float32)
    h2_np = h2.cpu().detach().numpy().astype(np.float32)

    # 数据有效性检查
    if h1_np.size == 0 or h2_np.size == 0:
        return 0.0

    # 过滤零方差特征
    valid_cols_h1 = np.where(np.std(h1_np, axis=0) > 1e-6)[0]
    valid_cols_h2 = np.where(np.std(h2_np, axis=0) > 1e-6)[0]
    h1_np = h1_np[:, valid_cols_h1]
    h2_np = h2_np[:, valid_cols_h2]

    # 样本量不足保护
    n_samples = h1_np.shape[0]
    if n_samples < 2 or h1_np.shape[1] == 0 or h2_np.shape[1] == 0:
        return 0.0

    # 自适应参数设置
    n_components = min(h1_np.shape[1], h2_np.shape[1], n_samples - 1)
    max_iter = max(10000, n_components * 200)  # 动态调整迭代次数

    # 标准化处理 (带平滑项)
    h1_np = (h1_np - np.mean(h1_np, axis=0)) / (np.std(h1_np, axis=0) + 1e-8)
    h2_np = (h2_np - np.mean(h2_np, axis=0)) / (np.std(h2_np, axis=0) + 1e-8)

    # 带异常捕获的CCA计算
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        try:
            cca = CCA(n_components=n_components, max_iter=max_iter)
            cca.fit(h1_np, h2_np)
            x, y = cca.transform(h1_np, h2_np)
            return float(np.mean([np.corrcoef(x[:, i], y[:, i])[0, 1]
                                  for i in range(n_components)]))
        except Exception as e:
            print(f"CCA计算失败: {str(e)}")
            return 0.0

def analyze_attention_patterns(model, batch):
    with torch.no_grad():
        outputs = model.bert.bert(batch['input_ids'], batch['attention_mask'], output_attentions=True)
        attentions = outputs.attentions  # Tuple of (layer_num, batch, head, seq_len, seq_len)

    # 计算各层注意力熵
    layer_entropy = []
    for layer_attn in attentions:
        attn = layer_attn.mean(dim=1)  # 平均多头注意力
        entropy = -torch.sum(attn * torch.log(attn + 1e-9), dim=-1).mean()
        layer_entropy.append(entropy.item())

    return {
        'attention_entropy': layer_entropy,
        'global_attention_var': torch.var(attentions[-1]).item()
    }


def feature_importance_analysis(model, sample):
    explainer = shap.DeepExplainer(model, sample)
    shap_values = explainer.shap_values(sample)

    importance_metrics = {
        "global_feat_imp": np.mean(np.abs(shap_values[0])),
        "local_feat_imp": np.mean(np.abs(shap_values[2]))
    }
    return importance_metrics


def track_feature_evolution(epoch_features):
    # epoch_features: List of dicts containing layer representations
    dynamics = {
        'global_velocity': float(np.linalg.norm(epoch_features[-1]['global'] - epoch_features[0]['global'])),
        'local_consistency': float(np.mean([cosine_similarity(
            torch.tensor(epoch_features[i]['local']).unsqueeze(0),
            torch.tensor(epoch_features[i + 1]['local']).unsqueeze(0)
        ).mean().item() for i in range(len(epoch_features) - 1)]))
    }
    return dynamics

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fast_mi_estimate(x, y, n_bins=64):
    """优化的二维直方图互信息计算（GPU加速）"""

    # 生成带保护边界的分箱边缘
    def _safe_edges(data):
        edges = torch.linspace(data.min(), data.max(), n_bins + 1, device=data.device)
        edges[-1] += 1e-6  # 防止最大值正好落在边缘上
        return edges

    # 离散化数据索引 (确保索引在[0, n_bins-1]范围内)
    x_edges = _safe_edges(x)
    x_idx = torch.bucketize(x, x_edges, right=False)
    x_idx = torch.clamp(x_idx, 0, n_bins - 1)

    y_edges = _safe_edges(y)
    y_idx = torch.bucketize(y, y_edges, right=False)
    y_idx = torch.clamp(y_idx, 0, n_bins - 1)

    # 创建联合直方图
    joint_hist = torch.zeros((n_bins, n_bins), device=x.device)
    joint_hist.index_put_(
        (x_idx, y_idx),
        torch.ones_like(x_idx, dtype=torch.float32),
        accumulate=True
    )

    # 计算概率分布
    joint_probs = joint_hist / joint_hist.sum()

    # 边缘概率分布
    marginal_x = joint_probs.sum(dim=1)
    marginal_y = joint_probs.sum(dim=0)

    # 计算互信息 (带数值稳定性处理)
    eps = 1e-9
    mi = torch.sum(joint_probs * (
            torch.log(joint_probs + eps) -
            torch.log(marginal_x[:, None] * marginal_y[None, :] + eps)
    ))

    return mi.item()


def batch_distance_corr(x, y, subsample=1024):
    """批量采样计算距离相关性"""
    idx = torch.randperm(x.size(0))[:subsample]
    return compute_distance_corr(x[idx], y[idx])


def fast_cca(x, y, pca_dim=32):
    """增强鲁棒性的CCA实现"""
    # 转换为numpy并移除NaN
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # 创建有效样本掩码
    valid_mask = ~(np.isnan(x_np).any(axis=1) | np.isnan(y_np).any(axis=1))

    # 至少保留50%样本
    if valid_mask.sum() < 0.5 * len(x_np):
        print(f"Warning: Over 50% samples contain NaN. Valid samples: {valid_mask.sum()}/{len(x_np)}")
        return 0.0

    x_clean = x_np[valid_mask]
    y_clean = y_np[valid_mask]

    # 确保有足够样本进行PCA
    min_samples = max(pca_dim * 2, 100)
    if len(x_clean) < min_samples:
        print(f"Insufficient samples ({len(x_clean)}) after NaN removal")
        return 0.0

    try:
        # PCA降维
        x_pca = PCA(n_components=pca_dim).fit_transform(x_clean)
        y_pca = PCA(n_components=pca_dim).fit_transform(y_clean)

        # CCA计算
        cca = CCA(n_components=1)
        cca.fit(x_pca, y_pca)
        x_c, y_c = cca.transform(x_pca, y_pca)

        # 计算相关系数
        corr = np.corrcoef(x_c.T, y_c.T)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    except Exception as e:
        print(f"CCA failed: {str(e)}")
        return 0.0


def track_feature_evolution(epoch_features):
    # epoch_features: List of dicts containing layer representations
    dynamics = {
        'global_velocity': float(np.linalg.norm(epoch_features[-1]['global'] - epoch_features[0]['global'])),
        'local_consistency': float(np.mean([cosine_similarity(
            torch.tensor(epoch_features[i]['local']).unsqueeze(0),
            torch.tensor(epoch_features[i + 1]['local']).unsqueeze(0)
        ).mean().item() for i in range(len(epoch_features) - 1)]))
    }
    return dynamics
