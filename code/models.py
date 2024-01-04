import torch
from torchvision import models, transforms
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights


class resnet_lstm(torch.nn.Module):
    def __init__(self, args, num_class):
        super(resnet_lstm, self).__init__()

        # self.save_hyperparameters()
        self.args = args


        # resnet = models.resnet50(pretrained=True)
        # self.res = torch.nn.Sequential()
        # self.res.add_module("conv1", resnet.conv1)
        # self.res.add_module("bn1", resnet.bn1)
        # self.res.add_module("relu", resnet.relu)
        # self.res.add_module("maxpool", resnet.maxpool)
        # self.res.add_module("layer1", resnet.layer1)
        # self.res.add_module("layer2", resnet.layer2)
        # self.res.add_module("layer3", resnet.layer3)
        # self.res.add_module("layer4", resnet.layer4)
        # self.res.add_module("avgpool", resnet.avgpool)
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        layers = list(backbone.children())[:-1]
        self.res = torch.nn.Sequential(*layers)

        self.lstm = torch.nn.LSTM(2048, 512, batch_first=True)
        self.fc = torch.nn.Linear(512, num_class)
        self.dropout = torch.nn.Dropout(p=0.2)

        torch.nn.init.xavier_normal_(self.lstm.all_weights[0][0])
        torch.nn.init.xavier_normal_(self.lstm.all_weights[0][1])
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        batch_size, sequence_length, channel, height, width = x.size()
        x = x.view(-1, sequence_length, channel, height, width)
        x = self.res.forward(x)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.dropout(y)
        y = self.fc(y)
        return y

    def get_optimizers(self):
        optimizer = None
        exp_lr_scheduler = None

        if self.args.opt == 0:
            optimizer = optim.SGD([
                {'params': self.res.parameters()},
                {'params': self.lstm.parameters(), 'lr': self.args.lr},
                {'params': self.fc.parameters(), 'lr': self.args.lr},
            ], lr=self.args.lr / 10, momentum=self.args.momentum, dampening=self.args.dampening,
                weight_decay=self.args.weightdecay, nesterov=self.args.nesterov)
            
        elif self.args.opt == 1:
            optimizer = optim.Adam([
                {'params': self.res.parameters()},
                {'params': self.lstm.parameters(), 'lr': self.args.lr},
                {'params': self.fc.parameters(), 'lr': self.args.lr},
            ], lr=self.args.lr / 10)

        return optimizer