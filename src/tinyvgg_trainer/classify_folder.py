"""
classify_folder.py  –  раскладывает изображения по подпапкам-классам
                      с помощью обученного TinyVGG.

✦ Пример:
    classify-tinyvgg

    python -m tinyvgg_trainer.classify_folder \
           --src data/to_sort \
           --dst data/sorted \
           --ckpt checkpoints/tinyvgg_best.pt \
           --classes-dir data/split/train \
           --img-size 64
"""

from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from tinyvgg_trainer.models.tiny_vgg import TinyVGG


# ──────────────────────────── util ──────────────────────────────
def discover_classes(classes_dir: Path) -> list[str]:
    """
    Читает список классов так же, как это делает torchvision.datasets.ImageFolder:
    алфавитный порядок подпапок.
    """
    if not classes_dir.is_dir():
        sys.exit(f"[Ошибка] classes-dir «{classes_dir}» не существует.")
    return sorted([p.name for p in classes_dir.iterdir() if p.is_dir()])


def load_model(ckpt: Path, num_classes: int, device: torch.device) -> TinyVGG:
    """Создаём TinyVGG и загружаем state-dict."""
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=num_classes)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model


@torch.inference_mode()
def classify_image(img_path: Path,
                   model: TinyVGG,
                   tfm: transforms.Compose,
                   device: torch.device,
                   class_names: list[str]) -> str:
    """Возвращает предсказанный класс для одного изображения."""
    img = Image.open(img_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0).to(device)  # [1, C, H, W]
    logits = model(tensor)
    pred_idx = logits.softmax(dim=1).argmax(dim=1).item()
    return class_names[pred_idx]


def main() -> None:
    path_src_dir = Path("src/tinyvgg_trainer/data/jpg_small_chank")
    path_weights_dir = Path("./weights")
    path_train_dir = Path("src/tinyvgg_trainer/data/split/train")
    path_sorted_dir = Path("src/tinyvgg_trainer/data/jpg_small_chank/sorted")
    path_sorted_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser("Классификация картинок и раскладка по папкам")
    parser.add_argument("--src", default=str(path_src_dir), help="Папка с неразмеченными картинками")
    parser.add_argument("--dst", default=str(path_sorted_dir), help="Корневая папка, куда складывать результат")
    parser.add_argument("--ckpt", required=True, help="Имя файла с весами TinyVGG (*.pt)")
    parser.add_argument("--classes-dir", default=str(path_train_dir),
                        help="Папка train с подпапками-классами (чтобы взять список классов). "
                             "Если не указана, классы читаются из dst (если уже есть) или ошибку.")
    parser.add_argument("--img-size", type=int, default=64, help="Размер, в который ресайзим изображения")
    parser.add_argument("--extensions", nargs="+",
                        default=[".jpg", ".jpeg", ".png", ".bmp"],
                        help="Какие расширения файлов считать изображениями")
    parser.add_argument("--copy", action="store_true",
                        help="Копировать файлы (по умолчанию перемещать)")

    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    ckpt = path_weights_dir / Path(args.ckpt)
    classes_dir = Path(args.classes_dir) if args.classes_dir else None

    if not src_dir.is_dir():
        sys.exit(f"[Ошибка] src «{src_dir}» не найдена")
    if not ckpt.is_file():
        sys.exit(f"[Ошибка] чекпоинт «{ckpt}» не найден")

    # ── определяем список классов ───────────────────────────────
    if classes_dir:
        class_names = discover_classes(classes_dir)
    elif dst_dir.exists():
        class_names = sorted([p.name for p in dst_dir.iterdir() if p.is_dir()])
    else:
        sys.exit("[Ошибка] Не удалось определить список классов: "
                 "укажите --classes-dir или создайте подпапки в dst.")

    print(f"Классы ({len(class_names)}): {class_names}")

    # ── устройство ───────────────────────────────────────────────
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Используем устройство: {device.type}")

    # ── трансформации (должны совпадать с обучением) ─────────────
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # ── загружаем сеть ───────────────────────────────────────────
    model = load_model(ckpt, num_classes=len(class_names), device=device)

    # ── обрабатываем файлы ───────────────────────────────────────
    img_paths = [p for p in src_dir.iterdir()
                 if p.suffix.lower() in args.extensions and p.is_file()]

    if not img_paths:
        sys.exit("[!] В папке src нет подходящих изображений")

    for img_path in tqdm(img_paths, desc="Классификация"):
        pred_class = classify_image(img_path, model, tfm, device, class_names)
        target_dir = dst_dir / pred_class
        target_dir.mkdir(parents=True, exist_ok=True)

        dst_file = target_dir / img_path.name
        if args.copy:
            shutil.copy2(str(img_path), str(dst_file))
        else:
            shutil.move(str(img_path), str(dst_file))

    print(f"Готово: обработано {len(img_paths)} изображений.")
    print(f"Результат лежит в: {dst_dir.resolve()}")


if __name__ == "__main__":
    main()