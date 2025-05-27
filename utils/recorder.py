from collections import defaultdict
from pathlib import Path
import json


class TrainingRecorder:
    def __init__(self):
        self.metrics = {
            'train': [],
            'val': [],
            'interpret': []
        }
        self.metadata = {}

    """增强型指标添加方法，支持动态结构验证"""

    def add_metrics(self, mode, metrics_dict):
        """增强型指标添加方法，支持动态结构验证"""
        # 结构验证
        if mode not in self.metrics:
            raise ValueError(f"Invalid mode: {mode}. Available modes: {list(self.metrics.keys())}")

        if mode == 'interpret':
            self._validate_interpret_metrics(metrics_dict)

        self.metrics[mode].append(metrics_dict)

    def _validate_interpret_metrics(self, metrics_dict):
        # 这里可以添加对解释性指标的结构验证逻辑
        pass

    """增强版保存方法，包含元数据"""

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

    """生成增强版可视化报告"""

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

    def _extract_trend(self, metric_name):
        # 提取指标的趋势数据
        return [entry[metric_name] for entry in self.metrics['interpret'] if metric_name in entry]

    def _extract_correlation(self, base_metric):
        # 提取相关性指标数据
        global_mid = [entry[f"{base_metric}_global-mid"] for entry in self.metrics['interpret'] if
                      f"{base_metric}_global-mid" in entry]
        mid_local = [entry[f"{base_metric}_mid-local"] for entry in self.metrics['interpret'] if
                     f"{base_metric}_mid-local" in entry]
        return {
            'global-mid': global_mid,
            'mid-local': mid_local
        }

    def _analyze_attention_patterns(self):
        # 分析注意力模式，这里可以添加具体逻辑
        return {}
