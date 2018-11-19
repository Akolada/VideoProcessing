import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
from instance_normalization_chainer.instance_normalization import InstanceNormalization

xp=cuda.cupy
cuda.get_device(0).use()

class CBR3D(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,activation=F.relu):
        w=initializers.Normal(0.01)
        self.up=up
        self.down=down
        self.activation=activation
        super(CBR3D,self).__init__()
        with self.init_scope():
            self.cpara=L.ConvolutionND(3,in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.ConvolutionND(3,in_ch,out_ch,4,2,1,initialW=w)
            self.cup=L.DeconvolutionND(3,in_ch,out_ch,4,2,1,initialW=w)

            self.bn0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h=self.activation(self.bn0(self.cup(x)))

        elif self.down:
            h=self.activation(self.bn0(self.cdown(x)))

        else:
            h=self.activation(self.bn0(self.cpara(x)))

        return h

class CBR2D(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,activation=F.relu):
        w=initializers.Normal(0.01)
        self.up=up
        self.down=down
        self.activation=activation
        super(CBR2D,self).__init__()
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

class Generator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(Generator,self).__init__()
        with self.init_scope():
            self.cbr3d_0=CBR3D(base*8,base*4,up=True)
            self.cbr3d_1=CBR3D(base*4,base*2,up=True)
            self.cbr3d_2=CBR3D(base*2,base,up=True)
            self.cbr3d_3=CBR3D(base,3,up=True)
            self.c_m=CBR3D(base,1,up=True)

            self.cbr2d_0=CBR2D(3,base,down=True)
            self.cbr2d_1=CBR2D(base,base*2,down=True)
            self.cbr2d_2=CBR2D(base*2,base*4,down=True)
            self.cbr2d_3=CBR2D(base*4,base*8,down=True)

            self.cbr2d_4=CBR2D(base*8,base*4,up=True)
            self.cbr2d_5=CBR2D(base*4,base*2,up=True)
            self.cbr2d_6=CBR2D(base*2,base,up=True)
            self.cbr2d_7=CBR2D(base,3,up=True)

    def encode(self,x):
        h=self.cbr2d_0(x)
        h=self.cbr2d_1(h)
        h=self.cbr2d_2(h)
        h=self.cbr2d_3(h)

        return h

    def foreground(self,x):
        x=x.reshape(2,512,1,8,8)
        h=self.cbr3d_0(x)
        h=self.cbr3d_1(h)
        hm=self.cbr3d_2(h)
        h=self.cbr3d_3(hm)
        mask=F.dropout(F.sigmoid(self.c_m(hm)),ratio=0.5)

        return F.tanh(h),mask

    def background(self,x):
        h=self.cbr2d_4(x)
        h=self.cbr2d_5(h)
        h=self.cbr2d_6(h)
        h=self.cbr2d_7(h)

        return F.tanh(h)

    def __call__(self,x):
        b,_,h,w=x.shape
        enc=self.encode(x)

        fg,mask=self.foreground(enc)
        mask=F.tile(mask,(1,3,1,1,1))

        bg=self.background(enc)
        bg=bg.reshape(b,3,1,h,w)
        bg=F.tile(bg,(1,1,16,1,1))

        return mask*fg + (1-mask)*bg

class Discriminator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.01)
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.cbr3d_0=CBR3D(3,base,down=True,activation=F.leaky_relu)
            self.cbr3d_1=CBR3D(base,base*2,down=True,activation=F.leaky_relu)
            self.cbr3d_2=CBR3D(base*2,base*4,down=True,activation=F.leaky_relu)
            self.cbr3d_3=CBR3D(base*4,base*8,down=True,activation=F.leaky_relu)
            self.cbr3d_4=L.Linear(None,1,initialW=w)

    def __call__(self,x):
        h=self.cbr3d_0(x)
        h=self.cbr3d_1(h)
        h=self.cbr3d_2(h)
        h=self.cbr3d_3(h)
        h=self.cbr3d_4(h)

        return h

class VGG(Chain):
    def __init__(self):
        super(VGG,self).__init__()
        with self.init_scope():
            self.base=L.VGG16Layers()

    def __call__(self,x):
        h=self.base(x,layers=["conv4_3"])["conv4_3"]

        return h