from torch import nn



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        )

    def forward(self, x):
        return self.layers(x)


class ConvNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=True):
        super().__init__()
        layers = [
            ConvLayer(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.branch = nn.Sequential(
            ConvNormLayer(in_channels, out_channels, kernel_size, 1),
            ConvNormLayer(out_channels, out_channels, kernel_size, 1, activation=False)
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x + self.branch(x)
        x = self.activation(x)
        return x


class ConvTanhLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormLayer(3, 24, 9, 1),
            ConvNormLayer(24, 48, 3, 2),
            ConvNormLayer(48, 96, 3, 2),
            ResLayer(96, 96, 3),
            ResLayer(96, 96, 3),
            ResLayer(96, 96, 3)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvNormLayer(96, 48, 3, 1),
            nn.Upsample(scale_factor=2),
            ConvNormLayer(48, 24, 3, 1),
            ConvTanhLayer(24, 3, 9, 1)
        )

    def forward(self, x):
        return self.layers(x)


class ReCoNetMin(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x