from pathlib import Path

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

"""
Создаёт последовательность преобразований. 
Все операции будут применяться по очереди к каждому изображению, когда оно загружается из датасета.
"""
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomRotation(degrees=(-90, 90)),
    transforms.ToTensor()
])

"""
Превращает данные нашего изображения в набор данных, который можно использовать с PyTorch
"""
path_split_data = Path("src/data/split")
path_train_data = path_split_data / "train"
path_test_data = path_split_data / "test"
path_val_data = path_split_data / "val"

train_data = datasets.ImageFolder(
    root=path_train_data,
    transform=data_transform,
    target_transform=None
)
test_data = datasets.ImageFolder(
    root=path_test_data,
    transform=data_transform,
    target_transform=None
)
val_data = datasets.ImageFolder(
    root=path_val_data,
    transform=data_transform,
    target_transform=None
)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}\nVal data:\n{val_data}")

"""
DataLoader используется во время обучения модели для пакетной загрузки данных с перемешиванием, 
параллельной обработкой и удобной итерацией, чтобы ускорить обучение модели и избежать переобучения.
"""
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=32,
                             num_workers=1,
                             shuffle=False) # don't usually need to shuffle testing data

val_dataloader = DataLoader(dataset=val_data,
                            batch_size=32,
                            num_workers=1,
                            shuffle=False)