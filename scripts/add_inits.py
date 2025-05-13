#!/usr/bin/env python3
import os
from pathlib import Path


def add_init_files():
    # Папка scripts/  → скрипт лежит в project_root/scripts/add_inits.py
    script_dir = Path(__file__).parent.resolve()
    # project_root — это родитель папки scripts/
    project_root = script_dir.parent
    # src/ теперь находится в корне проекта
    base_path = project_root / "src"

    if not base_path.exists():
        print(f"❌ Папка не найдена: {base_path}")
        return

    print(f"🔍 Сканируем директорию: {base_path}\n")
    for root, dirs, files in os.walk(base_path):
        print(f"— Зашли в: {root}")
        # создаём __init__.py в текущем каталоге
        init_in_root = Path(root) / "__init__.py"
        if not init_in_root.exists():
            init_in_root.write_text("# auto-generated __init__.py\n")
            print(f"    ✔ Создан {init_in_root}")
        # создаём __init__.py в каждой найденной подпапке
        for d in dirs:
            dir_path = Path(root) / d
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# auto-generated __init__.py\n")
                print(f"    ✔ Создан {init_file}")


if __name__ == "__main__":
    add_init_files()