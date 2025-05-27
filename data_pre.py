import json

import json
from pathlib import Path
from typing import Dict


def preprocess_labels(input_path: str, output_path: str = None) -> Dict:
    """
    标签预处理函数（增强版）
    功能：将所有标签类型映射到从0开始的连续整数

    Args:
        input_path: 原始数据文件路径
        output_path: 预处理后保存路径（可选）

    Returns:
        预处理后的数据字典（包含映射后的标签）
    """
    # 读取原始数据
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    # 对每个标签类型进行统一处理
    for label_type in ["global_labels", "intermediate_labels", "local_labels"]:
        if label_type not in raw_data:
            continue

        # 生成唯一值的有序列表（保证映射一致性）
        unique_values = sorted(set(raw_data[label_type]))

        # 创建映射字典（原始值 -> 新索引）
        value_mapping = {orig: idx for idx, orig in enumerate(unique_values)}

        # 应用映射转换
        raw_data[label_type] = [value_mapping[x] for x in raw_data[label_type]]

    # 验证标签合法性
    _validate_labels(raw_data)

    # 保存处理后的数据
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
        print(f"预处理数据已保存至：{output_path}")

    return raw_data


def _validate_labels(data: Dict):
    """增强版标签校验（基于实际映射后的取值范围）"""
    for label_type in ["global_labels", "intermediate_labels", "local_labels"]:
        labels = data.get(label_type, [])
        if not labels:  # 允许空标签列表
            continue

        # 计算当前标签的实际取值范围
        unique_values = set(labels)
        max_observed = max(unique_values) if unique_values else 0
        expected_max = len(unique_values) - 1 if unique_values else 0

        # 验证所有值都在合法范围内
        assert all(0 <= x <= expected_max for x in labels), (
            f"非法{label_type}值检测到！"
            f"实际范围0-{max_observed}，预期范围0-{expected_max}"
        )

        # 验证映射的连续性（可选）
        if sorted(unique_values) != list(range(expected_max + 1)):
            print(f"警告：{label_type}的值不连续，但仍在合法范围内")


if __name__ == "__main__":
    # 示例配置（按需修改）
    input_file = "ais_default_code_config_repo_441532/data/label_data_20000.txt"
    output_file = "ais_default_code_config_repo_441532/data/preprocessed_data_20k.txt"

    # 执行预处理
    processed_data = preprocess_labels(input_file, output_file)

    # 验证输出
    print("全局标签分布:", {
        "唯一值数量": len(set(processed_data["global_labels"])),
        "最大值": max(processed_data["global_labels"]) if processed_data["global_labels"] else None
    })
    print("局部标签分布:", {
        "唯一值数量": len(set(processed_data["intermediate_labels"])),
        "最大值": max(processed_data["intermediate_labels"]) if processed_data["intermediate_labels"] else None
    })
    print("全局标签分布:", {
        "唯一值数量": len(set(processed_data["local_labels"])),
        "最大值": max(processed_data["local_labels"]) if processed_data["local_labels"] else None
    })
