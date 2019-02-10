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

class CBR3D_dis(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,activation=F.relu):
        w=initializers.Normal(0.01)
        self.up=up
        self.down=down
        self.activation=activation
        super(CBR3D_dis,self).__init__()
        with self.init_scope():
            self.cpara=L.ConvolutionND(3,in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.ConvolutionND(3,in_ch,out_ch,4,(2,2,2),1,initialW=w)
            self.cup=L.DeconvolutionND(3,in_ch,out_ch,4,(2,2,2),1,initialW=w)

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

class ResBlock_2D(Chain):
    def __init__(self,in_ch,out_ch):
        super(ResBlock_2D,self).__init__()
        with self.init_scope():
            self.cbr0 = CBR2D(in_ch,out_ch)
            self.cbr1 = CBR2D(out_ch,out_ch)
    
    def __call__(self,x):
        h = self.cbr0(x)
        h = self.cbr1(h)

        return h+x

class Generator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.01)
        super(Generator,self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(128, 512*4*4, initialW=w)
            self.cbr3d_0=CBR3D(base*8,base*4,up=True)
            self.cbr3d_1=CBR3D(base*4,base*2,up=True)
            self.cbr3d_2=CBR3D(base*2,base,up=True)
            self.cbr3d_3=CBR3D(base,base,up=True)
            self.cbr3d_4 = L.Convolution3D(base,3,3,1,1,initialW=w)
            self.c_m=L.Convolution3D(base,1,3,1,1,initialW=w)

            self.cbr2d_0=CBR2D(3,base,down=True)
            self.cbr2d_1=CBR2D(base,base*2,down=True)
            self.cbr2d_2=CBR2D(base*2,base*4,down=True)
            self.cbr2d_3=CBR2D(base*4,base*8,down=True)
            self.res0 = ResBlock_2D(base*8, base*8)
            self.res1 = ResBlock_2D(base*8, base*8)

            self.cbr2d_4=CBR2D(base*8,base*4,up=True)
            self.cbr2d_5=CBR2D(base*4,base*2,up=True)
            self.cbr2d_6=CBR2D(base*2,base,up=True)
            self.cbr2d_7=CBR2D(base,base,up=True)
            self.cbr2d_8=L.Convolution2D(base,3,3,1,1,initialW=w)

    def encode(self,x):
        h=self.cbr2d_0(x)
        h1=self.cbr2d_1(h)
        h2=self.cbr2d_2(h1)
        h=self.cbr2d_3(h2)
        h=self.res0(h)
        h3=self.res1(h)

        return h1,h2,h3

    def foreground(self,x):
        x=x.reshape(2,512,1,4,4)
        h=self.cbr3d_0(x)
        h=self.cbr3d_1(h)
        h=self.cbr3d_2(h)
        hm=self.cbr3d_3(h)
        h=self.cbr3d_4(hm)
        mask=F.dropout(F.sigmoid(self.c_m(hm)),ratio=0.5)

        return F.tanh(h),mask

    def background(self,z):
        h=self.cbr2d_4(z)
        h=self.cbr2d_5(h)
        h=self.cbr2d_6(h)
        h=self.cbr2d_7(h)
        h=self.cbr2d_8(h)

        return F.tanh(h)

    def __call__(self,z):
        b = z.shape[0]
        h = 64
        w = 64
        #enc1,enc2,enc3=self.encode(x)

        z = F.relu(self.l0(z)).reshape(b, 512, 4, 4)

        fg,mask=self.foreground(z)
        mask=F.tile(mask,(1,3,1,1,1))

        bg=self.background(z)
        bg=bg.reshape(b,3,1,h,w)
        bg=F.tile(bg,(1,1,16,1,1))

        return mask*fg + (1-mask)*bg

class Discriminator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.01)
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.cbr3d_0=CBR3D_dis(3,base,down=True,activation=F.leaky_relu)
            self.cbr3d_01=CBR3D_dis(base,base,activation=F.leaky_relu)
            self.cbr3d_1=CBR3D_dis(base,base*2,down=True,activation=F.leaky_relu)
            self.cbr3d_11=CBR3D_dis(base*2,base*2,activation=F.leaky_relu)
            self.cbr3d_2=CBR3D_dis(base*2,base*4,down=True,activation=F.leaky_relu)
            self.cbr3d_21=CBR3D_dis(base*4,base*4,activation=F.leaky_relu)
            self.cbr3d_3=CBR3D_dis(base*4,base*8,down=True,activation=F.leaky_relu)
            self.cbr3d_adv=L.Linear(None,1,initialW=w)
            self.cbr3d_cls = L.Linear(None, 2,initialW=w)

    def __call__(self,x):
        h=self.cbr3d_0(x)
        h=self.cbr3d_01(h)
        h=self.cbr3d_1(h)
        h=self.cbr3d_11(h)
        h=self.cbr3d_2(h)
        h=self.cbr3d_21(h)
        hout=self.cbr3d_3(h)
        hadv=self.cbr3d_adv(hout)
        hcls=self.cbr3d_cls(hout)

        return hadv

class VGG(Chain):
    def __init__(self):
        super(VGG,self).__init__()
        with self.init_scope():
            self.base=L.VGG16Layers()

    def __call__(self,x):
        h=self.base(x,layers=["conv4_3"])["conv4_3"]

        return h