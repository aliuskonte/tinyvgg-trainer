"""
split_dataset.py — CLI для разбиения датасета на train/val/test.
Запуск:
    split-dataset [RAW_DATA_DIR]
По умолчанию берёт папку data/raw.
"""
import argparse
import os
from pathlib import Path

import splitfolders


def split_dataset(raw_data_dir: str):
    """
    Разбивает папку data/<raw> на train/val/test в data/split.
    :param raw_data_dir: имя папки с необработанными данными (внутри data/).
    """
    path_raw_data = Path("src/data") / raw_data_dir
    path_split_data = Path("src/data/split")
    path_split_data.mkdir(parents=True, exist_ok=True)

    # Вывод статистики по исходным директориям
    for sub in path_raw_data.iterdir():
        if sub.is_dir():
            count = len(list((path_raw_data / sub.name).iterdir()))
            print(f"{sub.name}: {count} файлов")

    # Разбиение
    splitfolders.ratio(
        input=str(path_raw_data),
        output=str(path_split_data),
        seed=1337,
        ratio=(0.8, 0.1, 0.1),
        group_prefix=None
    )
    print("Split completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Разбиение data/<raw> на train/val/test"
    )
    parser.add_argument(
        'raw_data_dir',
        nargs='?',
        default='raw',
        help='Имя папки внутри data (по умолчанию "raw")'
    )
    args = parser.parse_args()
    split_dataset(args.raw_data_dir)


if __name__ == "__main__":
    main()
