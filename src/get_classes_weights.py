import argparse

import pandas as pd
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        type=str,
        help="YAML file with configurations"
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    train_data = pd.read_csv(config.datasets.train[0].metadata_path)

    # use map to build "target" column
    train_data["target"] = train_data[config.datasets.train[0].target_column].map(
        config.data.label2id
    )

    # Calculate class weights
    class_counts = train_data["target"].value_counts().to_dict()
    total_samples = len(train_data)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    print("Class weights:", class_weights)

    # sort class weights by class id
    class_weights = {k: v for k, v in sorted(class_weights.items(), key=lambda item: item[0])}

    print("Class weights:", class_weights)

    # take only the values
    class_weights = list(class_weights.values())
    print("Class weights:", class_weights)

    print(config.data.label2id)
    print(train_data["target"].value_counts())


if __name__ == "__main__":
    main()