import os
from pathlib import Path

import splitfolders


def main(raw_data_dir):
    path_raw_data = Path(f"../data/{raw_data_dir}")
    path_split_data = Path("../data/split")
    path_split_data.mkdir(parents=True, exist_ok=True)

    print(os.listdir(path_raw_data/"hr"))
    print(len(os.listdir(path_raw_data/"hr")))

    train_dir = path_split_data/"train"
    test_dir = path_split_data/"test"

    splitfolders.ratio(path_raw_data, seed=1337, output=path_split_data, ratio=(0.8, 0.1, 0.1))

    # Проверка каталогов
    def walk_through_dir(output):
        for dirpath, dirnames, filenames in os.walk(output):
            print(f"Здесь {len(dirnames)} дирректории {len(filenames)} изображений в '{dirpath}'.")


    walk_through_dir(path_split_data)
    print("Готово!")


if __name__ == "__main__":

    main(raw_data_dir)