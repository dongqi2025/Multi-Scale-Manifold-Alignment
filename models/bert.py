import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import hashlib

"""改进的高斯KL散度计算，增强数值稳定性"""


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


class MSMA(nn.Module):
    def __init__(self, config):
        # 初始化所有组件
        super().__init__()
        self.config = config
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"self.device :{self.device}")

        # 模型组件
        self.bert = MultiscaleBert().to(self.device)
        # print(f"self.bert:{self.bert}")  # 查看MultiscaleBert结构中的归一化层

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
                h_local, h_mid, h_global = self.bert(batch['input_ids'], batch['attention_mask'])

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
        h_local, h_intermediate, h_global = self.bert(input_ids, attention_mask)

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
        h_local, h_intermediate, h_global = self.bert(input_ids, attention_mask)

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

    def run(self, resume: bool = False):
        self.best_val_loss = float('inf')
        # 新增：用于跟踪特征演化
        epoch_features_cache = []
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 0
        if resume:
            checkpoint_files = sorted(
                save_dir.glob("checkpoint_epoch*.pt"),
                key=lambda x: int(x.stem.split("_")[-1][5:])  # 提取epoch数
            )

            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]
                try:
                    self, start_epoch = MSMA.load_from_checkpoint(
                        str(latest_checkpoint),
                        device=self.device
                    )
                    print(f">>> 成功恢复训练，从epoch {start_epoch}开始")
                except Exception as e:
                    print(f"!!! 恢复失败: {str(e)}，将从头开始训练")
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

                # self.best_val_loss = val_loss
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
        os.system("cd /ossfs/workspace/checkpoints_20k/ && du -h -d 2")
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

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device=None):
        """从检查点加载完整训练状态

        Args:
            checkpoint_path: 检查点文件路径
            device: 目标设备 (自动检测)
        Returns:
            model: 恢复的模型实例
            epoch: 恢复的训练轮次
        """
        # 设备检测
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 初始化模型实例
        model = cls(checkpoint['config'])

        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复训练状态
        model.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        model.recorder.metrics = checkpoint.get('metrics', {
            'train': [], 'val': [], 'interpret': []
        })

        # 设备迁移
        model.to(device)

        return model, checkpoint['epoch']

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
                h_local, h_mid, h_global = self.bert(
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


"""检查嵌入数据是否已存在"""


def embeddings_exist(save_dir):
    """检查嵌入数据是否已存在"""
    emb_dir = Path(save_dir) / "embeddings"
    required_files = {'global.npy', 'intermediate.npy', 'local.npy', 'labels.npy'}
    existing_files = {f.name for f in emb_dir.glob('*') if f.is_file()}
    return required_files.issubset(existing_files)
