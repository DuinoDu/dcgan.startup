import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from itertools import izip


class GAN(object):

    """Base class for GAN"""

    def __init__(self):
        self.z_dims = 3
        self._nets = []
        self._criterions = []
        self._optims = []
        self._errors = []
        self._cuda = False
        self._phase = 'eval'

    def set_criterion(self, criterions):
        self._criterions = criterions

    def set_optim(self, optims):
        self._optims = optims

    def cuda(self):
        """set net to cuda mode.
        """
        self._cuda = True
        for net in self._nets:
            net.cuda()

    def train(self):
        """set net to train mode, just as pytorch does.
        """
        self._phase = 'train'
        for net in self._nets:
            net.train()

    def eval(self):
        """set net to eval mode, just as pytorch does.
        """
        self._phase = 'eval'
        for net in self._nets:
            net.eval()

    def parameters(self):
        """Return parameters of each net

        Returns: list of params

        """
        params = []
        for net in self._nets:
            params.append(net.parameters())
        return params

    def criterion(self):
        """Return errors by forward net

        Returns:
            errors(list): Variable
        """
        return self._errors

    def __call__(self, x):
        """forward each net, core of any derived GAN

        Args:
            x (Variable): input data

        Returns: 
            generated data

        """
        raise NotImplementedError	

    def backward_step(self):
        """ Backward all nets at one time.  
        """
        if len(self._optims) == 0:
            raise "please set optimizer for each net"

        for net, err, optim in izip(self._nets, self._errors, self._optims):
            net.zero_grad()
            err.backward()
            optim.step()

    def load_state_dicts(self, state_dicts):
        for net, state_dict in izip(self._nets, state_dicts):
            net.load_state_dict(state_dict)
        
    def state_dicts(self):
        states = []
        for net in self._nets:
            states.append(net.state_dict())
        return states


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, ngpu, nz, nc, ngf, ):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    

class BicycleGAN(GAN):
    """BicycleGAN
    https://arxiv.org/pdf/1711.11586.pdf
    """
    def __init__(self):
        super(DCGAN, self).__init__()
        self._nets.append(_netG())
        self._nets.append(_netD())
        self._nets.append(_netE())

    def __call__(self, x):
        if not isinstance(x, Variable):
            raise "input should be Variable"

        G = self._nets[0] # G --- 0
        D = self._nets[1] # D --- 1 
        E = self._nets[2] # E --- 2

        batch_size = x.data.size(0)
        z = torch.FloatTensor(0).resize_(batch_size, self.z_dims, 1, 1).normal_(0,1)
        label_real = torch.FloatTensor(batch_size).fill_(1)
        label_fake = torch.FloatTensor(batch_size).fill_(0)
        if self._cuda:
            x = x.cuda()
            z = z.cuda()
            label_real = label_real.cuda()
            label_fake = label_fake.cuda()
        z = Variable(z)
        label_real = Variable(label_real)
        label_fake = Variable(label_fake)
        fake = G(z) # generate fake

        # forward D
        if self._phase == 'train' and len(self._criterions) > 0:
            errD_real = self._criterions[1](D(x), label_real)
            errD_fake = self._criterions[1](D(fake.detach()), label_fake)
            errD = errD_real + errD_fake
            errG = self._criterions[0](D(fake), label_real)
            self._errors = [errG, errD]
        return fake


def test_D_G():
    from torch.autograd import Variable
    nz = 1
    nc = 3
    net = _netG(1, nz, nc, 64)
    x = Variable(torch.randn(1,nz,1,1))
    pred = net(x)
    print(pred.size())

    netD = _netD(1, nc, 64)
    pred2 = netD(pred)
    print(pred2.size())

def test_DCGAN():
    import os
    from torch.autograd import Variable
    import torch.optim as optim

    gpuID = 1
    os.environ['CUDA_VISIBLE_DIVICES'] = str(gpuID)
    torch.cuda.set_device(gpuID)

    # net
    net = DCGAN()
    # loss
    net.set_criterion([nn.BCELoss(), nn.BCELoss()])
    # optim
    params = net.parameters()
    optim_G = optim.Adam(params[0], lr=0.001, betas=(0.1, 0.999))
    optim_D = optim.Adam(params[1], lr=0.001, betas=(0.1, 0.999))
    net.set_optim([optim_G, optim_D])
    net.train()
    net.cuda()

    # forward
    batch_size = 32
    x = Variable(torch.randn(batch_size,3,64,64)).fill_(0)
    for i in range(1000):
        # backward & update
        pred = net(x)
        loss_G, loss_D = net.criterion()
        net.backward_step()

        #print(pred.size())
        vis = lambda x : x.data.cpu().numpy()[0]
        print("%d/1000 : loss_G: %.4f | loss_D: %.4f" % (i, vis(loss_G), vis(loss_D)))


if __name__ == "__main__":
    # test_D_G()
    test_DCGAN()
