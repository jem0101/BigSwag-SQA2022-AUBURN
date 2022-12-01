import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()

        self.ngf = kwargs["ngf"]
        self.z_dim = kwargs["z_dim"]
        self.n_channel = kwargs["n_channel"]
        self.image_size = kwargs["input_size"]

        main = nn.Sequential(
            # Z goes into a linear of size: self.ngf
            nn.Linear(self.z_dim, self.ngf),
            nn.ReLU(True),
            nn.Linear(self.ngf, self.ngf),
            nn.ReLU(True),
            nn.Linear(self.ngf, self.ngf),
            nn.ReLU(True),
            nn.Linear(self.ngf, self.n_channel * self.image_size * self.image_size),
            nn.Tanh(),
        )
        self.main = main

    def forward(self, z_input):
        z_input = z_input.view(z_input.size(0), z_input.size(1))
        output = self.main(z_input)
        return output.view(output.size(0), self.n_channel, self.image_size, self.image_size)


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()

        self.ndf = kwargs["ndf"]
        self.z_dim = kwargs["z_dim"]
        self.n_channel = kwargs["n_channel"]
        self.image_size = kwargs["input_size"]
        self.output_dimension = kwargs["output_disc"]
        main = nn.Sequential(
            # Z goes into a linear of size: self.ndf
            nn.Linear(self.n_channel * self.image_size * self.image_size, self.ndf),
            nn.ReLU(True),
            nn.Linear(self.ndf, self.ndf),
            nn.ReLU(True),
            nn.Linear(self.ndf, self.ndf),
            nn.ReLU(True),
            nn.Linear(self.ndf, self.output_dimension),
        )
        self.main = main

    def forward(self, input_data):
        input_data = input_data.view(
            input_data.shape[0], input_data.shape[1] * input_data.shape[2] * input_data.shape[3])
        output = self.main(input_data)
        ouput = output.view(input_data.shape[0], self.output_dimension)
        return ouput
