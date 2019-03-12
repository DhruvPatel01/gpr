import models
import torch
import dataset
import numpy as np

embedding_dim = 768

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.gap_model1(embedding_dim).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam([model.W, model.b])

train_data = dataset.DS_v1('data/pickles/dev/', 'data/embed/dev/')
val_data = dataset.DS_v1('data/pickles/val/', 'data/embed/val/')
train_data_size = len(train_data)
val_data_size = len(val_data)

epochs = 10
for ep in range(epochs):
    for i in range(train_data_size):
        x, y, p, _1, _2 = train_data[i]
        x, y, p = torch.tensor(x).to(device), torch.tensor([np.argmax(y)]).to(device), torch.tensor(p).unsqueeze(0).to(device)
        optimizer.zero_grad()
        y_ = model.forward(x, p)
        loss = criterion((y_.squeeze(1)).unsqueeze(0), y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch: {}, Step: {} -- Training Loss: {}".format(ep, i+1, loss))

    val_loss = 0
    for i in range(val_data_size):
        val_x, val_y, val_p, _1, _2 = val_data[i]
        val_x, val_y, val_p = torch.tensor(val_x).to(device), torch.tensor([np.argmax(val_y)]).to(device), torch.tensor(val_p).unsqueeze(0).to(device)
        val_y_ = model.forward(val_x, val_p)
        val_loss += criterion((val_y_.squeeze(1)).unsqueeze(0), val_y)
    print("***** Epoch: {} -- Validation Loss: {}".format(ep, val_loss/val_data_size))
