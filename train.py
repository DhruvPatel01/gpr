import models
import torch
## import dataset file

embedding_dim = 764

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.gap_model1(embedding_dim).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())

train_data = #Datasetmodule()
val_data = #Datasetmodule()
val_data_size = len(val_data)

epochs = 10
for ep in epochs:
    step = 1
    for x, y, p in data:
        y_ = model.forward(x.to(device), p.to(device))
        optimizer.zero_grad()
        loss = criterion(y.to(device), y_)
        loss.backward()
        optimizer.step()
        print("Epoch: {}, Step: {} -- Training Loss: {}".format(ep, step, loss))
        step += 1

    val_loss = 0
    for val_x, val_y, val_p in val_data:
        val_y_ = model.forward(val_x.to(device), val_p.to(device))
        val_loss += criterion(val_y.to(device), val_y_)
    print("***** Epoch: {} -- Validation Loss: {}".format(ep, val_loss/val_data_size))
