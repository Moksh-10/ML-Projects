# import timeit
#
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from model import vit
# from dataset import train_dl, val_dl
# from tqdm import tqdm
#
#
# LEARNING_RATE = 1e-4
# NUM_EPOCHS = 30
# LOAD_MODEL = False
#
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f'using deivce: {device}')
#
# # print(len(train_dl))
#
# def save_checkpoint(state, filename="checkpoint.pth.tar"):
#     print("====saving given checkpoint====")
#     torch.save(state, filename)
#
#
# def load_checkpoint(checkpoint, model, optimizer):
#     print("====loading checkpoint====")
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
#
#
# model = vit(emb_dim=16, patch_size=4, num_patches=4,
#             in_channels=3, num_enc=6, num_classes=101,
#             d_model=16, h=8, d_ff=2048)
#
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scaler = torch.cuda.amp.GradScaler()
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
#
# start = timeit.default_timer()
#
# for e in range(NUM_EPOCHS):
#
#     model.to(device)
#     model.train()
#
#     losses = []
#     train_labels = []
#     train_preds = []
#     train_r_loss = 0
#
#     loop = tqdm(train_dl)
#     for batch_idx, (data, target) in enumerate(loop):
#         x = data.float().to(device)
#         y = target.type(torch.uint8).to(device)
#         train_labels.extend(y.tolist())
#
#         with torch.cuda.amp.autocast():
#             y_pred = model(x)
#             y_pred_label = torch.argmax(y_pred, dim=1)
#             train_preds.extend(y_pred_label.tolist())
#
#             loss = loss_fn(y_pred, y)
#             losses.append(loss.item())
#             train_r_loss += loss.item()
#
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         # loss.backward()
#         scaler.step(optimizer)
#         # optimizer.step()
#
#         loop.set_postfix(loss=loss.item())
#
#     mean_loss = sum(losses) / len(losses)
#     mean_loss = round(mean_loss, 2)
#     scheduler.step(mean_loss)
#
#     train_loss = train_r_loss / (idx + 1)
#     train_acc = sum(1 for x, y in zip(train_preds, train_labels) if x==y) / len(train_labels)
#
#     model.eval()
#
#     val_labels = []
#     val_preds = []
#     val_r_loss = 0
#
#     l = tqdm(val_dl)
#     for batch_idx, (data, target) in enumerate(l):
#         x = data.float().to(device)
#         y = target.type(torch.uint8).to(device)
#         val_labels.extend(y.tolist())
#
#         with torch.cuda.amp.autocast():
#             y_pred = model(x)
#             y_pred_label = torch.argmax(y_pred, dim=1)
#             val_preds.extend(y_pred_label.tolist())
#
#             loss = loss_fn(y_pred, y)
#             losses.append(loss.item())
#             train_r_loss += loss.item()
#
#     val_loss = val_r_loss / (idx + 1)
#     val_acc = sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels)
#
#
#
#
#
