import functools
import math
import operator

import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):
    """
    - A VGG-style 3D CNN with 11 layers.
    - Kernel size is kept 3 for all three dimensions - (time, H, W)
      except the first layer has kernel size of (3, 5, 5)
    - Time dimension is preserved with `padding=1` and `stride=1`, and is
      averaged at the end

    Arguments:
    - Input: a (batch_size, 2, 3, sequence_length, W, H) tensor
    - Returns: a (batch_size, 512) sized tensor
    """

    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes

        # Load ResNet and replace last fc layer
        resnet = torchvision.models.resnet50(pretrained=True)
        # # 0: conv1
        # # 1: bn1
        # # 2: relu
        # # 3: maxpool
        # # 4: layer1
        # # 5: layer2
        # # 6: layer3
        # # 7: layer4
        # # 8: avgpool
        # # 9: fc
        # children = list(resnet.children())
        # self.resnet_features = nn.Sequential(
        #     children[0],
        #     children[1],
        #     children[2],
        #     children[3],
        #     children[4],
        #     children[5],
        #     children[6],
        #     children[8]
        # )
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        for layer in list(self.resnet_features.children())[:-3]:
            for param in layer.parameters():
                param.requires_grad = False
        self.mlp = nn.Sequential(nn.Linear(2*resnet.fc.in_features, math.floor(0.5 * resnet.fc.in_features)),
                                 nn.ReLU(),
                                 nn.Linear(math.floor(0.5 * resnet.fc.in_features), self.num_classes),
                                 nn.ReLU())
        # self.mlp = nn.Sequential(nn.Linear(2*resnet.fc.in_features, self.num_classes),
        #                          nn.ReLU())
        # self.resnet = torchvision.models.resnet50(pretrained=True)
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)
        # print(list(self.resnet.children()))

        # def hook(module, input, output):
        #     N,C,H,W = output.shape
        #     self.features = output.reshape(N,C,-1)
        # handle = self.resnet._modules.get('features')

    def forward(self, x, get_features=False):
        assert(len(x) == 1)
        # Reshape (batch, 2, 3, 224, 224) into (2*batch, 3, 224, 224)
        x = x[0].view(-1, *x[0].shape[2:])
        x = self.resnet_features(x)
        features = x

        # Flatten (2*batch, 2048, 1, 1) to (batch, 4096)
        x = x.view(-1, 2 * functools.reduce(operator.mul, x.shape[1:], 1))
        x = self.mlp(x)

        if get_features:
            return x, features
        return x


if __name__ == "__main__":
    num_classes = 174
    input_tensor = torch.autograd.Variable(torch.rand(5, 3, 72, 84, 84))
    model = Model(512).cuda()

    output = model(input_tensor.cuda())
    print(output.size())
