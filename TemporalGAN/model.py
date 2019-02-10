import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np

class CBR_1D(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,depthwise=False, activation=F.relu):
        w = initializers.Normal(0.01)
        self.up = up
        self.down = down
        self.depthwise = depthwise
        self.activation = activation
        super(CBR_1D,self).__init__()
        with self.init_scope():
            self.cpara = L.ConvolutionND(1,in_ch,out_ch,3,1,1,initialW=w)
            self.cdown = L.ConvolutionND(1,in_ch,out_ch,4,2,1,initialW=w)
            self.cdw = L.Convolution1D(in_ch, out_ch, 1,1,0,initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h = F.unpooling_nd(x,2,2,0,cover_all=False)
            h = self.activation(self.bn0(self.cpara(h)))

        elif self.down:
            h = self.activation(self.bn0(self.cdown(x)))

        elif self.depthwise:
            h = self.activation(self.bn0(self.cdw(x)))

        else:
            h = self.activation(self.bn0(self.cpara(x)))

        return h

class TemporalGenerator(Chain):
    def __init__(self,base=64):
        w = initializers.Normal(0.01)
        super(TemporalGenerator,self).__init__()
        with self.init_scope():
            self.c0 = CBR_1D(base*2, base*8, depthwise=True)
            self.c1 = CBR_1D(base*8, base*4, up=True)
            self.c2 = CBR_1D(base*4, base*2, up=True)
            self.c3 = CBR_1D(base*2, base*2, up=True)
            self.c4 = L.Deconvolution1D(base*2, base*2, 4,2,1,initialW=w)

    def __call__(self,x):
        h = x.reshape(x.shape[0],-1,1)
        h = self.c0(h)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)

        return F.tanh(h)

class CBR_2D(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,activation=F.relu):
        w = initializers.Normal(0.01)
        self.up = up
        self.down = down
        self.activation = activation
        super(CBR_2D,self).__init__()
        with self.init_scope():
            self.cpara = L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.cdown = L.Convolution2D(in_ch,out_ch,4,2,1,initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h = F.unpooling_2d(x,2,2,0,cover_all=False)
            h = self.activation(self.bn0(self.cpara(h)))

        elif self.down:
            h = self.activation(self.bn0(self.cdown(x)))

        else:
            h = self.activation(self.bn0(self.cpara(x)))

        return h

class ImageGenerator(Chain):
    def __init__(self,base=64):
        super(ImageGenerator,self).__init__()
        w = initializers.Normal(0.01)
        with self.init_scope():
            self.l0 = L.Linear(base*2, 4*4*base*4, initialW=w)
            self.l1 = L.Linear(base*2*4*4, 4*4*base*4, initialW=w)
            self.c0 = CBR_2D(base*8, base*4, up=True)
            self.c1 = CBR_2D(base*4, base*2, up=True)
            self.c2 = CBR_2D(base*2, base, up=True)
            self.c3 = CBR_2D(base,base,up=True)
            self.c4 = L.Convolution2D(base,3,3,1,1,initialW=w)

            self.bn0 = L.BatchNormalization(4*4*base*4)
            self.bn1 = L.BatchNormalization(4*4*base*4)

    def __call__(self,x,temp):
        b = x.shape[0]
        h0 = F.relu(self.bn0(self.l0(x))).reshape(b,256,4,4)
        h1 = F.relu(self.bn1(self.l1(temp))).reshape(b,256,4,4)
        h = self.c0(F.concat([h0,h1]))
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)

        return F.tanh(h)

class CBR_3D(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,activation=F.relu):
        w = initializers.Normal(0.01)
        self.up = up
        self.down = down
        self.activation = activation
        super(CBR_3D,self).__init__()
        with self.init_scope():
            self.cpara = L.ConvolutionND(3,in_ch,out_ch,3,1,1,initialW=w)
            self.cdown = L.ConvolutionND(3,in_ch,out_ch,4,2,1,initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h = F.unpooling_nd(x,2,2,0,cover_all=False)
            h = self.activation(self.bn0(self.cpara(h)))

        elif self.down:
            h = self.activation(self.bn0(self.cdown(x)))
            #h = self.activation(self.cdown(x))

        else:
            h = self.activation(self.bn0(self.cpara(x)))

        return h

class Discriminator(Chain):
    def __init__(self,base=64):
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.c0 = CBR_3D(3,base,down=True,activation=F.leaky_relu)
            self.c1 = CBR_3D(base,base*2,down=True,activation=F.leaky_relu)
            self.c2 = CBR_3D(base*2, base*4,down=True,activation=F.leaky_relu)
            self.c3 = CBR_3D(base*4, base*8,down=True,activation=F.leaky_relu)
            self.c4 = L.Convolution2D(base*8,1,4,1,1,initialW=initializers.Normal(0.01))

    def __call__(self,x):
        h = self.c0(x)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = F.reshape(h, (h.shape[0] * h.shape[2],) + self.c4.W.shape[1:])
        h = self.c4(h)

        return h