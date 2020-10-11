import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.InstanceNorm2d,
                 use_bias=True, scale_factor=1):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * scale_factor, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=use_bias),
            norm_layer(in_channels * scale_factor),
            nn.Conv2d(in_channels=in_channels * scale_factor, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=use_bias),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            SeparableConv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, use_bias=True),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            SeparableConv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, use_bias=True),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class generator_encoder_Small(nn.Module):

    def __init__(self, conv_dim=64, repeat_num=6, input_nc=3, multi=4):
        super(generator_encoder_Small, self).__init__()
        conv_dim_s = int(conv_dim/multi)
        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim_s, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim_s, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_0_0 = nn.Sequential(*layers_branch)
        layers_branch = []
        layers_branch.append(nn.Conv2d(conv_dim_s, conv_dim_s * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim_s * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_0_1 = nn.Sequential(*layers_branch)

        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim_s, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim_s, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_1_0 = nn.Sequential(*layers_branch)
        layers_branch = []
        layers_branch.append(nn.Conv2d(conv_dim_s, conv_dim_s * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim_s * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_1_1 = nn.Sequential(*layers_branch)

        self.conv11_x_0 = nn.Conv2d(conv_dim_s, conv_dim, 1, 1, 0)
        self.conv11_x_1 = nn.Conv2d(conv_dim_s*2, conv_dim*2, 1, 1, 0)
        self.conv11_y_0 = nn.Conv2d(conv_dim_s, conv_dim, 1, 1, 0)
        self.conv11_y_1 = nn.Conv2d(conv_dim_s * 2, conv_dim * 2, 1, 1, 0)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, y):
        input_x_0 = self.Branch_0_0(x)
        input_x_1 = self.Branch_0_1(input_x_0)
        input_y_0 = self.Branch_1_0(y)
        input_y_1 = self.Branch_1_1(input_y_0)
        out_x_0 = self.relu(self.conv11_x_0(input_x_0))
        out_x_1 = self.relu(self.conv11_x_1(input_x_1))
        out_y_0 = self.relu(self.conv11_y_0(input_y_0))
        out_y_1 = self.relu(self.conv11_y_1(input_y_1))
        out = torch.cat((out_x_1, out_y_1), dim=1)
        res = [out_x_0, out_x_1, out_y_0, out_y_1, out]
        return res


class Big_encoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6, input_nc=3):
        super(Big_encoder, self).__init__()

        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_0_0 = nn.Sequential(*layers_branch)
        layers_branch = []
        layers_branch.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_0_1 = nn.Sequential(*layers_branch)

        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_1_0 = nn.Sequential(*layers_branch)
        layers_branch = []
        layers_branch.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_1_1 = nn.Sequential(*layers_branch)

    def forward(self, x, y):
        input_x_0 = self.Branch_0_0(x)
        input_x_1 = self.Branch_0_1(input_x_0)
        input_y_0 = self.Branch_1_0(y)
        input_y_1 = self.Branch_1_1(input_y_0)
        return [input_x_0, input_x_1, input_y_0, input_y_1]


class Big_decoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=9, input_nc=3):
        super(Big_decoder, self).__init__()
        # Down-Sampling, branch merge
        layers = []
        curr_dim = conv_dim * 2 #128           #256
        layers.append(nn.Conv2d(curr_dim * 2, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2  #256

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers_1 = []
        layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.Tanh())
        self.branch_1 = nn.Sequential(*layers_1)
        layers_2 = []
        layers_2.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_2.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        layers_2.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_2.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        layers_2.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_2.append(nn.Tanh())
        self.branch_2 = nn.Sequential(*layers_2)
    def forward(self, x):
        out = self.main(x)
        out_A = self.branch_1(out)
        out_B = self.branch_2(out)
        return out_A, out_B

class generator(nn.Module):

    def __init__(self, conv_dim=64, repeat_num=6, input_nc=3):
        super(generator, self).__init__()

        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        layers_branch.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_0 = nn.Sequential(*layers_branch)

        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        layers_branch.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_1 = nn.Sequential(*layers_branch)

        # Down-Sampling,  merge
        layers = []
        curr_dim = conv_dim * 2 #128           #256
        layers.append(nn.Conv2d(curr_dim * 2, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2  #256

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers_1 = []
        layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.Tanh())
        self.branch_1 = nn.Sequential(*layers_1)
        layers_2 = []
        layers_2.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_2.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        layers_2.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_2.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        layers_2.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_2.append(nn.Tanh())
        self.branch_2 = nn.Sequential(*layers_2)
    def forward(self, x, y):
        input_x = self.Branch_0(x)
        input_y = self.Branch_1(y)
        input_fuse = torch.cat((input_x, input_y), dim=1)
        out = self.main(input_fuse)
        out_A = self.branch_1(out)
        out_B = self.branch_2(out)
        return out_A, out_B


from spectral_norm import spectral_norm as SpectralNorm
class discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, repeat_num=3):
        super(discriminator, self).__init__()
        layers = []

        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for _ in range(1, repeat_num):

            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        #k_size = int(image_size / np.power(2, repeat_num))

        layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, padding=1)))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim *2

        self.main = nn.Sequential(*layers)
        self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))


        # conv1 remain the last square size, 256*256-->30*30
        #self.conv2 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False))
        #conv2 output a single number

    def forward(self, x):
        h = self.main(x)
        #out_real = self.conv1(h)
        out_makeup = self.conv1(h)
        #return out_real.squeeze(), out_makeup.squeeze()
        return out_makeup.squeeze()
