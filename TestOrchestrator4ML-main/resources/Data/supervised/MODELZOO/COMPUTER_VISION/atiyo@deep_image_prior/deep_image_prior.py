import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from PIL import Image


#use cuda, or not? be prepared for a long wait if you don't have cuda capabilities.
use_cuda = True
#input image. the architectures have been designed for 512x512 colour images
ground_truth_path = 'bunny_512.jpg'
#proportion of pixels to black out.
prop = 0.5
#standard deviation of added noise after each training set
sigma = 1./30
#number of training iterations
num_steps = 25001
#number of steps to take before saving an output image
save_frequency = 250
#where to put the output
output_name = 'output/output'
#choose either 'pixel_shuffle' or 'deconv' as the architecture used.
method = 'pixel_shuffle'

#accept a file path to a jpg, return a torch tensor
def jpg_to_tensor(filepath=ground_truth_path):
    pil = Image.open(ground_truth_path)
    pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if use_cuda:
        tensor = pil_to_tensor(pil).cuda()
    else:
        tensor = pil_to_tensor(pil)
    return tensor.view([1]+list(tensor.shape))

#accept a torch tensor, convert it to a jpg at a certain path
def tensor_to_jpg(tensor, filename):
    tensor = tensor.view(tensor.shape[1:])
    if use_cuda:
        tensor = tensor.cpu()
    tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = tensor_to_pil(tensor)
    pil.save(filename)

#function which zeros out a random proportion of pixels from an image tensor.
def zero_out_pixels(tensor, prop=prop):
    if use_cuda:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:])).cuda()
    else:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:]))
    mask[mask<prop] = 0
    mask[mask!=0] = 1
    mask = mask.repeat(1,3,1,1)
    deconstructed = tensor * mask
    return mask, deconstructed

#define an encoder decoder network with pixel shuffle upsampling
class pixel_shuffle_hourglass(nn.Module):
    def __init__(self):
        super(pixel_shuffle_hourglass, self).__init__()
        self.d_conv_1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.d_bn_1 = nn.BatchNorm2d(8)

        self.d_conv_2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.d_bn_2 = nn.BatchNorm2d(16)

        self.d_conv_3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.d_bn_3 = nn.BatchNorm2d(32)
        self.s_conv_3 = nn.Conv2d(32, 4, 5, stride=1, padding=2)

        self.d_conv_4 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.d_bn_4 = nn.BatchNorm2d(64)
        self.s_conv_4 = nn.Conv2d(64, 4, 5, stride=1, padding=2)

        self.d_conv_5 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.d_bn_5 = nn.BatchNorm2d(128)
        self.s_conv_5 = nn.Conv2d(128, 4, 5, stride=1, padding=2)

        self.d_conv_6 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.d_bn_6 = nn.BatchNorm2d(256)

        self.u_conv_5 = nn.Conv2d(68, 128, 5, stride=1, padding=2)
        self.u_bn_5 = nn.BatchNorm2d(128)

        self.u_conv_4 = nn.Conv2d(36, 64, 5, stride=1, padding=2)
        self.u_bn_4 = nn.BatchNorm2d(64)

        self.u_conv_3 = nn.Conv2d(20, 32, 5, stride=1, padding=2)
        self.u_bn_3 = nn.BatchNorm2d(32)

        self.u_conv_2 = nn.Conv2d(8, 16, 5, stride=1, padding=2)
        self.u_bn_2 = nn.BatchNorm2d(16)

        self.u_conv_1 = nn.Conv2d(4, 16, 5, stride=1, padding=2)
        self.u_bn_1 = nn.BatchNorm2d(16)

        self.out_conv = nn.Conv2d(4, 3, 5, stride=1, padding=2)
        self.out_bn = nn.BatchNorm2d(3)

        
    def forward(self, noise):
        down_1 = self.d_conv_1(noise)
        down_1 = self.d_bn_1(down_1)
        down_1 = F.leaky_relu(down_1)
        
        down_2 = self.d_conv_2(down_1)
        down_2 = self.d_bn_2(down_2)
        down_2 = F.leaky_relu(down_2)

        down_3 = self.d_conv_3(down_2)
        down_3 = self.d_bn_3(down_3)
        down_3 = F.leaky_relu(down_3)
        skip_3 = self.s_conv_3(down_3)

        down_4 = self.d_conv_4(down_3)
        down_4 = self.d_bn_4(down_4)
        down_4 = F.leaky_relu(down_4)
        skip_4 = self.s_conv_4(down_4)

        down_5 = self.d_conv_5(down_4)
        down_5 = self.d_bn_5(down_5)
        down_5 = F.leaky_relu(down_5)
        skip_5 = self.s_conv_5(down_5)

        down_6 = self.d_conv_6(down_5)
        down_6 = self.d_bn_6(down_6)
        down_6 = F.leaky_relu(down_6)

        up_5 = F.pixel_shuffle(down_6, 2)
        up_5 = torch.cat([up_5, skip_5], 1)
        up_5 = self.u_conv_5(up_5)
        up_5 = self.u_bn_5(up_5)
        up_5 = F.leaky_relu(up_5)

        up_4 = F.pixel_shuffle(up_5, 2)
        up_4 = torch.cat([up_4, skip_4], 1)
        up_4 = self.u_conv_4(up_4)
        up_4 = self.u_bn_4(up_4)
        up_4 = F.leaky_relu(up_4)

        up_3 = F.pixel_shuffle(up_4, 2)
        up_3 = torch.cat([up_3, skip_3], 1)
        up_3 = self.u_conv_3(up_3)
        up_3 = self.u_bn_3(up_3)
        up_3 = F.leaky_relu(up_3)

        up_2 = F.pixel_shuffle(up_3, 2)
        up_2 = self.u_conv_2(up_2)
        up_2 = self.u_bn_2(up_2)
        up_2 = F.leaky_relu(up_2)

        up_1 = F.pixel_shuffle(up_2, 2)
        up_1 = self.u_conv_1(up_1)
        up_1 = self.u_bn_1(up_1)
        up_1 = F.leaky_relu(up_1)

        out = F.pixel_shuffle(up_1, 2)
        out = self.out_conv(out)
        out = self.out_bn(out)
        out = F.sigmoid(out)
        return out

#define an encoder decoder network with convolution transpose upsampling.
class deconv_hourglass(nn.Module):
    def __init__(self):
        super(deconv_hourglass, self).__init__()
        self.d_conv_1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.d_bn_1 = nn.BatchNorm2d(8)

        self.d_conv_2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.d_bn_2 = nn.BatchNorm2d(16)

        self.d_conv_3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.d_bn_3 = nn.BatchNorm2d(32)
        self.s_conv_3 = nn.Conv2d(32, 4, 5, stride=1, padding=2)

        self.d_conv_4 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.d_bn_4 = nn.BatchNorm2d(64)
        self.s_conv_4 = nn.Conv2d(64, 4, 5, stride=1, padding=2)

        self.d_conv_5 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.d_bn_5 = nn.BatchNorm2d(128)
        self.s_conv_5 = nn.Conv2d(128, 4, 5, stride=1, padding=2)

        self.d_conv_6 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.d_bn_6 = nn.BatchNorm2d(256)

        self.u_deconv_5 = nn.ConvTranspose2d(256, 124, 4, stride=2, padding=1)
        self.u_bn_5 = nn.BatchNorm2d(128)

        self.u_deconv_4 = nn.ConvTranspose2d(128, 60, 4, stride=2, padding=1)
        self.u_bn_4 = nn.BatchNorm2d(64)

        self.u_deconv_3 = nn.ConvTranspose2d(64, 28, 4, stride=2, padding=1)
        self.u_bn_3 = nn.BatchNorm2d(32)

        self.u_deconv_2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.u_bn_2 = nn.BatchNorm2d(16)

        self.u_deconv_1 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.u_bn_1 = nn.BatchNorm2d(8)

        self.out_deconv = nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)        
        self.out_bn = nn.BatchNorm2d(3)

        
    def forward(self, noise):
        down_1 = self.d_conv_1(noise)
        down_1 = self.d_bn_1(down_1)
        down_1 = F.leaky_relu(down_1)
        
        down_2 = self.d_conv_2(down_1)
        down_2 = self.d_bn_2(down_2)
        down_2 = F.leaky_relu(down_2)

        down_3 = self.d_conv_3(down_2)
        down_3 = self.d_bn_3(down_3)
        down_3 = F.leaky_relu(down_3)
        skip_3 = self.s_conv_3(down_3)

        down_4 = self.d_conv_4(down_3)
        down_4 = self.d_bn_4(down_4)
        down_4 = F.leaky_relu(down_4)
        skip_4 = self.s_conv_4(down_4)

        down_5 = self.d_conv_5(down_4)
        down_5 = self.d_bn_5(down_5)
        down_5 = F.leaky_relu(down_5)
        skip_5 = self.s_conv_5(down_5)

        down_6 = self.d_conv_6(down_5)
        down_6 = self.d_bn_6(down_6)
        down_6 = F.leaky_relu(down_6)

        up_5 = self.u_deconv_5(down_6)
        up_5 = torch.cat([up_5, skip_5], 1)
        up_5 = self.u_bn_5(up_5)
        up_5 = F.leaky_relu(up_5)

        up_4 = self.u_deconv_4(up_5)
        up_4 = torch.cat([up_4, skip_4], 1)
        up_4 = self.u_bn_4(up_4)
        up_4 = F.leaky_relu(up_4)

        up_3 = self.u_deconv_3(up_4)
        up_3 = torch.cat([up_3, skip_3], 1)
        up_3 = self.u_bn_3(up_3)
        up_3 = F.leaky_relu(up_3)

        up_2 = self.u_deconv_2(up_3)
        up_2 = self.u_bn_2(up_2)
        up_2 = F.leaky_relu(up_2)

        up_1 = self.u_deconv_1(up_2)
        up_1 = self.u_bn_1(up_1)
        up_1 = F.leaky_relu(up_1)

        out = self.out_deconv(up_1)
        out = self.out_bn(out)
        out = F.sigmoid(out)

        return out

if __name__=='__main__':
    #import image
    truth = jpg_to_tensor(ground_truth_path)
    #deconstruct image
    mask, deconstructed = zero_out_pixels(truth)
    #save the deconstructed image
    tensor_to_jpg(deconstructed, 'deconstructed.jpg')
    #convert the image and mask to variables.
    mask = Variable(mask)
    deconstructed = Variable(deconstructed)

    #input of the network is noise
    if use_cuda:
        noise = Variable(torch.randn(deconstructed.shape).cuda())
    else:
        noise = Variable(torch.randn(deconstructed.shape))

    #initialise the network with the chosen architecture
    if method=='pixel_shuffle':
        net = pixel_shuffle_hourglass()
    elif method=='deconv':
        net = deconv_hourglass()

    #bind the network to the gpu if cuda is enabled
    if use_cuda:
        net.cuda()
    #network optimizer set up
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    #dummy index to provide names to output files
    save_img_ind = 0
    for step in range(num_steps):
        #get the network output
        output = net(noise)
        #we are only concerned with the output where we have the image available.
        masked_output = output * mask
        # calculate the l2_loss over the masked output and take an optimizer step
        optimizer.zero_grad()
        loss = torch.sum((masked_output - deconstructed)**2)
        loss.backward()
        optimizer.step()
        print('At step {}, loss is {}'.format(step, loss.data.cpu()))
        #every save_frequency steps, save a jpg
        if step % save_frequency == 0:
            tensor_to_jpg(output.data,output_name+'_{}.jpg'.format(save_img_ind))
            save_img_ind += 1
        if use_cuda:
            noise.data += sigma * torch.randn(noise.shape).cuda()
        else:
            noise.data += sigma * torch.randn(noise.shape)

    #clean up any mess we're leaving on the gpu
    if use_cuda:
        torch.cuda.empty_cache()
