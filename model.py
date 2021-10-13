import torch
from torch import nn, optim

class FurnitureClassifierV1(nn.Module):
    """
    Our standard model that classifies the 3 classes of furniture items.
    The model size can be modified. Additionally, dropout helps prevent
    overfitting (our dataset is quite small even with combative measures)
    but increases the amount of time to train the model.
    """
    def __init__(self, n=4, out=3, f=128, dropout=0.4):
        super(FurnitureClassifierV1, self).__init__()
        self.n = n
        self.out = out
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n, kernel_size=7, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=n*2, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=n*2, out_channels=n*4, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels=n*4, out_channels=n*8, kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(in_features=n*8*16*16, out_features=f)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=f, out_features=out)

    def forward(self, inp):
        x = inp
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        out = self.fc2(torch.relu(self.drop(self.fc1(x.flatten(1, 3)))))
        return out

    def evaluate(self, inp):
        """
        Used in getting prediction classes from the model in a
        more production-friendly method. Use this method with
        a wrapper when releasing a production model. Note that
        input preprocessing needs to be handled by the wrapper
        as well.
        """
        x = torch.unsqueeze(inp, 0)
        out = self.forward(x)
        return out.max(1)[1].item()

    
