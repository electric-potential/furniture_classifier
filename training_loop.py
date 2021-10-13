import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import time
import model
import data_preprocessing
import pickle
import os

def get_accuracy(mod, dataloader):
    """
    Returns number of correct and incorrect predictions
    for each class in the model. Use this to evaluate the
    prediction accuracy on the validation set to try and
    detect overfitting.
    """
    mod.eval()
    n = len(dataloader)
    correct = [0]*mod.out
    incorrect = [0]*mod.out

    with torch.no_grad():
        for batch in dataloader:
            out = mod(batch[0]).max(1)[1]
            for i in range(len(out)):
                label = batch[1][i].item()
                pred = out[i].item()
                if label == pred:
                    correct[label] += 1
                else:
                    incorrect[label] += 1
    return correct, incorrect

def train_model(mod, t_dataloader, v_dataloader, lr = 0.001, decay=0.05, epochs=100, start_epoch=0, checkpoint_path="./checkpoints/", print_sched=1, save_sched=5):
    """
    The training loop. Checkpoints will be stored for loading with
    relevant information in the directory.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(mod.parameters(), lr=lr, weight_decay=decay)
    optimizer.zero_grad()
    mod.train()

    if start_epoch == 0:
        t_losses, t_cors, t_incors, v_losses, v_cors, v_incors = [], [], [], [], [], []
    else:
        with open(os.path.join(checkpoint_path, "t_losses.pk"), "rb") as f:
            t_losses = pickle.load(f)
        with open(os.path.join(checkpoint_path, "v_losses.pk"), "rb") as f:
            v_losses = pickle.load(f)
        with open(os.path.join(checkpoint_path, "t_cors.pk"), "rb") as f:
            t_cors = pickle.load(f)
        with open(os.path.join(checkpoint_path, "t_incors.pk"), "rb") as f:
            t_incors = pickle.load(f)
        with open(os.path.join(checkpoint_path, "v_cors.pk"), "rb") as f:
            v_cors = pickle.load(f)
        with open(os.path.join(checkpoint_path, "v_incors.pk"), "rb") as f:
            v_incors = pickle.load(f)

    start = time.time()
    last = start
    print("")

    for epoch in range(start_epoch, start_epoch+epochs):
        t_loss = 0
        for batch in t_dataloader:
            xs = batch[0]
            gt = nn.functional.one_hot(batch[1], num_classes=mod.out).type_as(xs)

            zs = mod(xs)
            loss = criterion(zs, gt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            t_loss += loss.item()/len(batch[0])

        v_loss = 0
        with torch.no_grad():
            for batch in t_dataloader:
                xs = batch[0]
                gt = nn.functional.one_hot(batch[1]).type_as(xs)

                zs = mod(xs)
                loss = criterion(zs, gt)

                v_loss += loss.item()/len(batch[0])

        t_cor, t_incor = get_accuracy(mod, t_dataloader)
        v_cor, v_incor = get_accuracy(mod, v_dataloader)
        mod.train()
        t_losses.append(t_loss)
        v_losses.append(v_loss)
        t_cors.append(t_cor)
        t_incors.append(t_incor)
        v_cors.append(v_cor)
        v_incors.append(v_incor)

        now = time.time()

        if print_sched is not None and epoch%print_sched == 0:
            print("epoch %d. [Train Loss %f, Valid Loss %f, Train Acc %.3f%%, Valid Acc %.3f%%]" % (
                epoch, t_losses[-1], v_losses[-1], 100*sum(t_cors[-1])/(sum(t_cors[-1])+sum(t_incors[-1])), 100*sum(v_cors[-1])/(sum(v_cors[-1])+sum(v_incors[-1]))))
            print("last iteration: %.3f second(s)" % (now-last))
            print("elapsed time: %.3f second(s)" % (now-start))
            print("")
        last = now

        if checkpoint_path is not None and save_sched is not None and epoch%save_sched == 0:
            torch.save(mod.state_dict(), os.path.join(checkpoint_path, "ckpt-epoch{}.pk").format(epoch))
            with open(os.path.join(checkpoint_path, "t_losses.pk"), "wb") as f:
                pickle.dump(t_losses, f)
            with open(os.path.join(checkpoint_path, "v_losses.pk"), "wb") as f:
                pickle.dump(v_losses, f)
            with open(os.path.join(checkpoint_path, "t_cors.pk"), "wb") as f:
                pickle.dump(t_cors, f)
            with open(os.path.join(checkpoint_path, "t_incors.pk"), "wb") as f:
                pickle.dump(t_incors, f)
            with open(os.path.join(checkpoint_path, "v_cors.pk"), "wb") as f:
                pickle.dump(v_cors, f)
            with open(os.path.join(checkpoint_path, "v_incors.pk"), "wb") as f:
                pickle.dump(v_incors, f)

    return t_losses, v_losses, t_cors, t_incors, v_cors, v_incors

def plot_training_curves(t_losses, v_losses, t_cors, t_incors, v_cors, v_incors, path="./checkpoints/"):
    """
    Common plotting utilities for losses and accuracies reported
    during training.
    """
    plt.title("Learning Curve: Loss per Epoch")
    plt.plot(t_losses, label="Train")
    plt.plot(v_losses, label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(path, "loss.png"))

    plt.clf()

    plt.title("Learning Curve: Prediction Accuracy per Epoch")
    plt.plot([sum(t_cors[i])/(sum(t_cors[i])+sum(t_incors[i])) for i in range(len(t_cors))], label="Train")
    plt.plot([sum(v_cors[i])/(sum(v_cors[i])+sum(v_incors[i])) for i in range(len(v_cors))], label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(path, "accuracy.png"))

    plt.clf()
    return
        
if __name__ == "__main__":
    d = data_preprocessing.FurnitureDataset("train_labels.csv", "./dataset/")
    m = model.FurnitureClassifierV1()
    t, v = torch.utils.data.random_split(d, [len(d)-(len(d)//10), len(d)//10])
    t = torch.utils.data.DataLoader(t, batch_size=64, shuffle=True)
    v = torch.utils.data.DataLoader(v, batch_size=64, shuffle=True)
    t_losses, v_losses, t_cors, t_incors, v_cors, v_incors = train_model(m, t, v, epochs=20, start_epoch=30)
    
