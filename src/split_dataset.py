import os
import argparse
from pathlib import Path

import splitfolders


def main(raw_data_dir: str):
    # Определяем пути
    path_raw_data = Path("data") / raw_data_dir
    path_split_data = Path("data/split")

    # Список имён всех поддиректорий в path_raw_data
    dir_names = [p.name for p in path_raw_data.iterdir() if p.is_dir()]

    # Создаем выходную папку, если её нет
    path_split_data.mkdir(parents=True, exist_ok=True)

    # Выводим информацию об исходных данных
    for dir_name in dir_names:
        dir_path = path_raw_data / dir_name
        print(f"Содержимое '{dir_path}'")
        print(f"Всего файлов: {len(os.listdir(dir_path))}\n")

    # Сплит данных в папки train, val, test
    splitfolders.ratio(
        input=str(path_raw_data),
        output=str(path_split_data),
        seed=1337,
        ratio=(0.8, 0.1, 0.1),
        group_prefix=None  # если нужно разбить без учета группировки
    )

    # Функция для проверки содержимого выходных папок
    def walk_through_dir(output: Path):
        for dirpath, dirnames, filenames in os.walk(output):
            print(f"В '{dirpath}' найдено {len(dirnames)} папок и {len(filenames)} файлов.")

    # Выводим итоги
    walk_through_dir(path_split_data)
    print("Готово!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Скрипт для разбиения дата-сета на train/val/test"
    )
    parser.add_argument(
        'raw_data_dir',
        nargs='?',
        default='raw',
        help='Имя папки с необработанными данными внутри ../data (по умолчанию "raw")'
    )
    args = parser.parse_args()
    main(args.raw_data_dir)
