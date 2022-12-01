#!/usr/bin/env python
# encoding: utf-8

import torch.nn as nn


def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, input_size, n_channel, k=100, ndf=64):
        super(Encoder, self).__init__()
        assert input_size % 16 == 0, "isize has to be a multiple of 16"

        # input is nc x isize x isize
        main = nn.Sequential()
        main.add_module('initial_conv_{0}-{1}'.format(n_channel, ndf),
                        nn.Conv2d(n_channel, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = input_size / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Conv2d(cndf, k, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input_data):
        output = self.main(input_data)
        return output


# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, input_size, n_channel, k=100, ngf=64):
        super(Decoder, self).__init__()
        assert input_size % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != input_size:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_convt'.format(k, cngf), nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf), nn.ReLU(True))

        csize = 4
        while csize < input_size // 2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid_{0}_relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final_{0}-{1}_convt'.format(cngf, n_channel),
                        nn.ConvTranspose2d(cngf, n_channel, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(n_channel),
                        nn.Tanh())

        self.main = main

    def forward(self, input_data):
        output = self.main(input_data)
        return output


# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        self.image_size = kwargs["input_size"][0]
        self.n_channel = kwargs["n_channel"]
        self.z_dim = kwargs["z_dim"]
        self.ngf = kwargs["ngf"]
        self.decoder = Decoder(self.image_size, n_channel=self.n_channel, k=self.z_dim, ngf=self.ngf)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.1)
                m.bias.data.fill_(0)

    def forward(self, z_input):
        z_input = z_input.view(z_input.shape[0], self.z_dim, 1, 1)
        output = self.decoder(z_input)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size

class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.output_decoder = kwargs["output_decoder"]
        self.image_size = kwargs["input_size"]
        self.n_channel = kwargs["n_channel"]
        self.z_dim = kwargs["z_dim"]
        self.output_dimension = kwargs["output_disc"]
        self.ndf = kwargs["ndf"]
        self.ngf = kwargs["ngf"]

        self.encoder = Encoder(self.image_size, n_channel=self.n_channel, k=self.output_dimension, ndf=self.ndf)
        self.decoder = Decoder(self.image_size, n_channel=self.n_channel, k=self.z_dim, ngf=self.ngf)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.1)
                m.bias.data.fill_(0)

    def forward(self, input_data):
        f_enc_X = self.encoder(input_data)
        if not self.output_decoder:
            return f_enc_X.view(input_data.shape[0], self.output_dimension)
        else:
            f_dec_X = self.decoder(f_enc_X)
            f_dec_X = f_dec_X.view(input_data.shape[0], self.n_channel, self.image_size, self.image_size)
            return f_enc_X.view(input_data.shape[0], self.output_dimension), f_dec_X
