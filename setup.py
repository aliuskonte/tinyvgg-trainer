# setup.py
# ------------------------------------------------------------------------------
# Установка пакета src:
#   pip install -e .
# После этого будут доступны команды:
#   train-tinyvgg     — обучение модели
#   split-dataset     — разбиение датасета на train/val/test
# ------------------------------------------------------------------------------

from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf‑8") if (here / "README.md").exists() else ""

setup(
    name="src",
    version="0.1.0",
    description="Небольшой пакет для обучения TinyVGG (train / val / test) + сплит датасета",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dmitry Migunov",
    python_requires=">=3.9",
    # -------------------------------------------------------------------------
    # Структура: исходники находятся в каталоге src/
    # Пакеты (каталоги с __init__.py)
    #   • src/models
    #   • src/utils
    # Модули‑скрипты (одиночные .py без __init__.py)
    #   • src/cli.py
    #   • src/prepare_dataloaders.py
    #   • src/split_dataset.py
    #   • src/training_loop.py
    # -------------------------------------------------------------------------
    package_dir={"": "src"}, # говорит, что все пакеты лежат внутри src/.
    packages=find_packages("src"), # найдёт tinyvgg_trainer и его подпакеты
    py_modules=[
        "cli",
        "prepare_dataloaders",
        "split_dataset",
        "training_loop",
    ],
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tqdm>=4.0.0",
        "split_folders>=0.5.1",
    ],
    entry_points={
        "console_scripts": [
            "train-tinyvgg=tinyvgg_trainer.cli:main",
            "split-dataset=tinyvgg_trainer.split_dataset:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)