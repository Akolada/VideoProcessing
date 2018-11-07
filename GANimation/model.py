import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np

xp=cuda.cupy
cuda.get_device(0).use()

class CBR(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,activation=F.relu):
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

        elif self.down:
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

class Predictor(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.c0=CBR(9,base)
            self.down0=CBR(base,base*2,down=True)
            self.down1=CBR(base*2,base*4,down=True)
            self.down2=CBR(base*4,base*8,down=True)
            self.down3=CBR(base*8,base*16,down=True)
            self.res0=ResBlock(base*16,base*16)
            self.res1=ResBlock(base*16,base*16)
            self.res2=ResBlock(base*16,base*16)
            self.res3=ResBlock(base*16,base*16)
            self.res4=ResBlock(base*16,base*16)
            self.res5=ResBlock(base*16,base*16)
            self.up0=CBR(base*32,base*8,up=True)
            self.up1=CBR(base*16,base*4,up=True)
            self.up2=CBR(base*8,base*2,up=True)
            self.up3=CBR(base*4,base,up=True)
            self.c1=L.Convolution2D(base*2,3,3,1,1,initialW=w)

    def __call__(self,x):
        h1=self.c0(x)
        h2=self.down0(h1)
        h3=self.down1(h2)
        h4=self.down2(h3)
        h5=self.down3(h4)
        h=self.res0(h5)
        h=self.res1(h)
        h=self.res2(h)
        h=self.res3(h)
        h=self.res4(h)
        h=self.res5(H)
        h=self.up0(F.concat([h5,h]))
        h=self.up1(F.concat([h4,h]))
        h=self.up2(F.concat([h3,h]))
        h=self.up3(F.concat([h2,h]))
        h=self.c1(F.concat([h1,h]))

        return F.tanh(h)

class Discriminator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.down0=CBR(6,base,down=True)
            self.down1=CBR(base,base*2,down=True)
            self.down2=CBR(base*2,base*4,down=True)
            self.down3=CBR(base*4,base*8,down=True)
            self.cout=L.Convolution2D(base*8,1,3,1,1,initialW=w)

    def __call__(self,x):
        h=self.cbr0(x)
        h=self.cbr1(h)
        h=self.cbr2(h)
        h=self.cbr3(h)
        h=self.cout(h)

        return h

