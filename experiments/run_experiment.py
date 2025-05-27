import yaml
from models.bert import MSMA as BERT_MSMA
from models.gpt2 import MSMA as GPT2_MSMA
import argparse


def main(config):
    model_type = config.get('model_type', 'bert')
    if model_type == 'bert':
        model = BERT_MSMA(config)
    elif model_type == 'gpt2':
        model = GPT2_MSMA(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to the config file")
    args = parser.parse_args()

    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
