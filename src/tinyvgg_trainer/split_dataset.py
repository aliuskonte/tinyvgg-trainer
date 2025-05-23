"""
split_dataset.py — CLI для разбиения датасета на train/val/test.
Запуск:
    split-dataset

    # классическое разбиение 80 / 10 / 10
    split-dataset data_raw

    # своё соотношение train-val-test
    split-dataset data_raw --ratio 0.7 0.2 0.1

    # только train-val (без test)
    split-dataset data_raw --ratio 0.85 0.15
По умолчанию берёт папку data/raw.
"""
import argparse
from pathlib import Path

import splitfolders


def split_dataset(raw_data_dir: str, ratio: tuple[float, ...]):
    """
    Разбивает папку data/<raw> на train/val/test в data/split.
    :param raw_data_dir: имя папки с необработанными данными (внутри data/).
    """
    path_raw_data = Path("src/tinyvgg_trainer/data") / raw_data_dir
    path_split_data = Path("src/tinyvgg_trainer/data/split")
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
        ratio=ratio,
        group_prefix=None
    )
    print("Split completed.")


def parse_ratio(values: list[str]) -> tuple[float, ...]:
    nums = tuple(float(v) for v in values)
    if len(nums) not in (2, 3):
        raise argparse.ArgumentTypeError("Нужно передать 2 или 3 числа.")
    if abs(sum(nums) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError("Сумма ratio должна быть 1.0.")
    return nums


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
    parser.add_argument(
        "--ratio",
        nargs="+",
        type=str,
        default=["0.8", "0.1", "0.1"],
        metavar="RATIO",
        help="Соотношение (2 или 3 числа, сумма = 1.0). "
             'Напр.: --ratio 0.75 0.25  или  --ratio 0.7 0.2 0.1',
    )
    args = parser.parse_args()

    try:
        ratio = parse_ratio(args.ratio)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    split_dataset(
        args.raw_data_dir,
        ratio
    )


if __name__ == "__main__":
    main()
