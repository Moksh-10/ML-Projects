import os

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def train():
    if global_rank == 0:
        initialize_services()  # wandb, etc

    data_loader = DataLoader(
        train_ds, shuffle=False, sample=DistributedSampler(train_ds, shuffle=True)
    )
    model = model()
    if os.path.exists("latest_checkpoint.pth"):
        model.load_state_dict(torch.load("latest_checkpoint.pth"))

    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-4, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss()

    for e in range(num_epochs):
        for x, y in data_loader:
            loss = loss_fn(model(data), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if global_rank == 0:
                collect_statistics()

        if global_rank == 0:
            torch.save(model.state_dict(), "latest_checkpoint.pth")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    train()

    destroy_process_group()
