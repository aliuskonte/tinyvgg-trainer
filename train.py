import torch
from torch import nn

from timeit import default_timer as timer

from src.models.tiny_vgg import TinyVGG
from src.prepare_dataloaders import train_dataloader, val_dataloader, test_dataloader, train_data
from src.training_loop import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 25

# Recreate an instance of TinyVGG
model = TinyVGG(
    input_shape=3,  # number of color channels (3 for RGB)
    hidden_units=10,
    output_shape=len(train_data.classes)
).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Start the timer
start_time = timer()

# Train model
model_results = train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS
)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")