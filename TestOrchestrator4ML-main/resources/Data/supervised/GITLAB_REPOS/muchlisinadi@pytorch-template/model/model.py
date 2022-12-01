import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from resource.networks import resnet
from resource.networks import densenet
from resource.networks import mlp
from resource.networks import encoder_decoder
from resource.networks import densenet_noise

class ResnetModel(BaseModel):
    def __init__(self, **kwargs):
        super(ResnetModel, self).__init__()
        self.generator = resnet.Generator(**kwargs)
        self.discriminator = resnet.Discriminator(**kwargs)

    def forward(self, z_input):
        return self.generator(z_input)

class MLPModel(BaseModel):
    def __init__(self, **kwargs):
        super(MLPModel, self).__init__()
        self.generator = mlp.Generator(**kwargs)
        self.discriminator = mlp.Discriminator(**kwargs)

    def forward(self, z_input):
        return self.generator(z_input)

class DensenetModel(BaseModel):
    def __init__(self, **kwargs):
        super(DensenetModel, self).__init__()
        self.generator = densenet.Generator(**kwargs)
        self.discriminator = densenet.Discriminator(**kwargs)

    def forward(self, z_input):
        return self.generator(z_input)

class DensenetNoiseModel(BaseModel):
    def __init__(self, **kwargs):
        super(DensenetNoiseModel, self).__init__()
        self.generator = densenet_noise.Generator(**kwargs)
        self.discriminator = densenet_noise.Discriminator(**kwargs)

    def forward(self, z_input):
        return self.generator(z_input)

class EncDecModel(BaseModel):
    def __init__(self, **kwargs):
        super(EncDecModel, self).__init__()
        self.generator = encoder_decoder.Generator(**kwargs)
        self.discriminator = encoder_decoder.Discriminator(**kwargs)

    def forward(self, z_input):
        return self.generator(z_input)
