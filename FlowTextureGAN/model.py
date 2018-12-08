import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np

xp=cuda.cupy
cuda.get_device(0).use()

class CBR_2D(Chain):
    def __init__(self,in_ch,out_ch,down=False,up=False,activation=F.relu):
        w=initializers.Normal(0.02)
        self.up=up
        self.down=down
        self.activation=activation
        super(CBR_2D,self).__init__()
        with self.init_scope():
            self.cpara=L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.Convolution2D(in_ch,out_ch,4,2,1,initialW=w)

            self.bn0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h=F.unpooling_2d(x,2,2,0,cover_all=False)
            h=self.activation(self.bn0(self.cpara(h)))

        elif self.down:
            h=self.activation(self.bn0(self.cdown(x)))

        else:
            h=self.activation(self.bn0(self.cpara(x)))

        return h

class CBR_3D(Chain):
    def __init__(self,in_ch,out_ch,down=False,up=False,activation=F.relu):
        w=initializers.Normal(0.01)
        self.up=up
        self.down=down
        self.activation=activation
        super(CBR_3D,self).__init__()
        with self.init_scope():
            self.cpara=L.ConvolutionND(3,in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.ConvolutionND(3,in_ch,out_ch,4,2,1,initialW=w)

            self.bn0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h=F.unpooling_nd(x,2,2,0,cover_all=False)
            h=self.activation(self.bn0(self.cpara(h)))

        elif self.down:
            h=self.activation(self.bn0(self.cdown(x)))

        else:
            h=self.activation(self.bn0(self.cpara(x)))

        return h

class Encoder(Chain):
    def __init__(self,base=64):
        super(Encoder,self).__init__()
        with self.init_scope():
            self.cbr0=CBR_2D(6,base,down=True)
            self.cbr1=CBR_2D(base,base*2,down=True)
            self.cbr2=CBR_2D(base*2,base*4,down=True)
            self.cbr3=CBR_2D(base*4,base*8,down=True)
            self.cbr4=CBR_2D(base*16,base*4,up=True)
            self.cbr5=CBR_2D(base*8,base*2,up=True)
            self.cbr6=CBR_2D(base*4,base,up=True)
            self.cbr7=CBR_2D(base*2,base,up=True)
            self.cbr8=L.Convolution2D(base,3,3,1,1,initialW=initializers.Normal(0.01))

    def __call__(self,x):
        h1=self.cbr0(x)
        h2=self.cbr1(h1)
        h3=self.cbr2(h2)
        h4=self.cbr3(h3)
        h=self.cbr4(F.concat([h4,h4]))
        h=self.cbr5(F.concat([h3,h]))
        h=self.cbr6(F.concat([h2,h]))
        h=self.cbr7(F.concat([h1,h]))
        h=self.cbr8(h)

        return F.tanh(h)

class Refine(Chain):
    def __init__(self,base=64):
        super(Refine,self).__init__()
        with self.init_scope():
            self.cbr0=CBR_3D(3,base,down=True)
            self.cbr1=CBR_3D(base,base*2,down=True)
            self.cbr2=CBR_3D(base*2,base*4,down=True)
            self.cbr3=CBR_3D(base*4,base*8,down=True)
            self.cbr4=CBR_3D(base*16,base*4,up=True)
            self.cbr5=CBR_3D(base*8,base*2,up=True)
            self.cbr6=CBR_3D(base*4,base,up=True)
            self.cbr7=CBR_3D(base*2,base,up=True)
            self.cbr8=L.ConvolutionND(3,base,3,3,1,1,initialW=initializers.Normal(0.01))

    def __call__(self,x):
        h1=self.cbr0(x)
        h2=self.cbr1(h1)
        h3=self.cbr2(h2)
        h4=self.cbr3(h3)
        h=self.cbr4(F.concat([h4,h4]))
        h=self.cbr5(F.concat([h3,h3]))
        h=self.cbr6(F.concat([h2,h]))
        h=self.cbr7(F.concat([h1,h]))
        h=self.cbr8(h)

        return F.tanh(h)

class Discriminator(Chain):
    def __init__(self,base=64):
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.cbr0=CBR_3D(3,base,down=True)
            self.cbr1=CBR_3D(base,base)
            self.cbr2=CBR_3D(base,base*2,down=True)
            self.cbr3=CBR_3D(base*2,base*2)
            self.cbr4=CBR_3D(base*2,base*4,down=True)
            self.cbr5=CBR_3D(base*4,base*8,down=True)
            self.l0=L.Linear(None,1)

    def __call__(self,x):
        h=self.cbr0(x)
        h=self.cbr1(h)
        h=self.cbr2(h)
        h=self.cbr3(h)
        h=self.cbr4(h)
        h=self.cbr5(h)
        h=self.l0(h)

        return h