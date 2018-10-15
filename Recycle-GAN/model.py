import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np

xp=cuda.cupy
cuda.get_device(0).use()

class CBR(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,self.activation=F.relu):
        super(CBR,self).__init__()
        w=initializers.Normal(0.02)
        self.up=up
        self.down=down
        self.activation=activation
        with self.init_scope():
            self.cpara=L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.Convolution2D(in_ch,out_ch,4,2,1,initialW=w)

            self.bn0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h=F.unpooling_2d(x,2,2,0,cover_all=False)
            h=self.activation(self.bn0(self.cpara(h)))

        if self.down:
            h=self.activation(self.bn0(self.cdown(h)))

        else:
            h=self.activation(self.bn0(self.cpara(h)))

        return h

class ResBlock(Chain):
    def __init__(self,in_ch,out_ch):
        super(ResBlock,self).__init__()
        with self.init_scope():
            self.cbr0=CBR(in_ch,out_ch)
            self.cbr1=CBR(out_ch,out_ch)

    def __call__(self,x):
        h=self.cbr0(x)
        h=self.cbr1(h)

        return h+x

class Generator(Chain):
    def __init__(self,base=32):
        super(Generator,self).__init__()
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.c0=L.Convolution2D(3,base,7,1,3,initialW=w)
            self.cbr0=CBR(base,base*2,down=True)
            self.cbr1=CBR(base*2,base*4,down=True)
            self.res0=ResBlock(base*4,base*4)
            self.res1=ResBlock(base*4,base*4)
            self.res2=ResBlock(base*4,base*4)
            self.res3=ResBlock(base*4,base*4)
            self.res4=ResBlock(base*4,base*4)
            self.res5=ResBlock(base*4,base*4)
            self.cbr2=CBR(base*4,base*2,up=True)
            self.cbr3=CBR(base*2,base,up=True)
            self.c1=L.Convolution2D(base,3,7,1,3,initialW=w)

            self.bn0=L.BatchNormalization(base)

    def __call__(self,x):
        h=F.relu(self.bn0(self.c0(x)))
        h=self.cbr0(h)
        h=self.cbr1(h)
        h=self.res0(h)
        h=self.res1(h)
        h=self.res2(h)
        h=self.res3(h)
        h=self.res4(h)
        h=self.res5(h)
        h=self.cbr2(h)
        h=self.cbr3(h)
        h=F.tanh(self.c1(h))

        return h

class Discriminator(Chain):
    def __init__(self,base=64):
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.cbr0=CBR(3,base,down=True,down=F.leaky_relu)
            self.cbr1=CBR(base,base*2,down=True,down=F.leaky_relu)
            self.cbr2=CBR(base*2,base*4,down=True,down=F.leaky_relu)
            self.cbr3=CBR(base*4,base*8,down=True,down=F.leaky_relu)
            self.cout=L.Convolution(base*8,1,3,1,1,initialW=w)

    def __call__(self,x):
        h=self.cbr0(x)
        h=self.cbr1(h)
        h=self.cbr2(h)
        h=self.cbr3(h)
        h=self.cout(h)

        return h

class UNet(Chain):
    def __init__(self,base=64):
        super(UNet,self).__init__()
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.c0=L.Convolution2D(3,base,3,1,1,initialW=w)
            self.cbr0=CBR(base,base*2,down=True)
            self.cbr1=CBR(base*2,base*4,down=True)
            self.cbr2=CBR(base*4,base*8,down=True)
            self.cbr3=CBR(base*8,base*8,down=True)
            self.cbr4=CBR(base*16,base*4,up=True)
            self.cbr5=CBR(base*8,base*2,down=True)
            self.cbr6=CBR(base*4,base*1,down=True)
            self.cbr7=CBR(base*2,base*1,down=True)
            self.c1=L.Convolution2D(base,3,3,1,1,initialW=w)

            self.bn0=L.BatchNormaliztion(base)

    def __call__(self,x):
        h1=F.relu(self.bn0(self.c0(x)))
        h2=self.cbr0(h)
        h3=self.cbr1(h)
        h4=self.cbr2(h)
        h5=self.cbr3(h)
        h=self.cbr4(F.concat([h5,h4]))
        h=self.cbr5(F.concat([h3,h]))
        h=self.cbr6(F.concat([h2,h]))
        h=self.cbr7(F.concat([h1,h]))
        h=self.c1(h)

        return F.tanh(h)
