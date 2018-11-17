import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers

xp=cuda.cupy
cuda.get_device(0).use()

class CBR3D(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False):
        w=initializers.Normal(0.01)
        self.up=up
        self.down=down
        super(CBR,self).__init__()
        with self.init_scope():
            self.cpara=L.ConvolutionND(ndim=3,in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.ConvolutionND(ndim=3,in_ch,out_ch,4,2,1,initialW=w)

            self.bn0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h=F.unpooling_nd(x,(2,2,2),2,0,cover_all=False)
            h=F.relu(self.bn0(self.cpara(h)))

        elif self.down:
            h=F.leaky_relu(self.bn0(self.cdown(x)))

        else:
            h=F.relu(self.bn0(self.cpara(x)))

        return h

class CBR2D(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False):
        w=initializers.Normal(0.01)
        self.up=up
        self.down=down
        super(CBR,self).__init__()
        with self.init_scope():
            self.cpara=L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.Convolution2D(in_ch,out_ch,4,2,1,initialW=w)

            self.bn0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h=F.unpooling_2d(x,2,2,0,cover_all=False)
            h=F.relu(self.bn0(self.cpara(h)))

        elif self.down:
            h=F.leaky_relu(self.bn0(self.cdown(x)))

        else:
            h=F.relu(self.bn0(self.cpara(x)))

class Generator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(Generator,self).__init__()
        with self.init_scope():
            self.cbr3d_0=CBR3D(3,base,down=True)
            self.cbr3d_1=CBR3D(base,base*2,down=True)
            self.cbr3d_2=CBR3D(base*2,base*4,down=True)
            self.cbr3d_3=CBR3D(base*4,base*8,down=True)
            self.cbr3d_4=CBR3D(base*16,base*4,up=True)
            self.cbr3d_5=CBR3D(base*8,base*2,up=True)
            self.cbr3d_6=CBR3D(base*4,base,up=True)
            self.cbr3d_7=CBR3D(base*2,3,up=True)
            self.c_m=CBR3D(base,1,up=True)

            self.cbr2d_0=CBR2D(3,base,down=True)
            self.cbr2d_1=CBR2D(base,base*2,down=True)
            self.cbr2d_2=CBR2D(base*2,base*4,down=True)
            self.cbr2d_3=CBR2D(base*4,base*8,down=True)
            self.cbr2d_4=CBR2D(base*16,,base*4,up=True)
            self.cbr2d_5=CBR2D(base*8,base*2,up=True)
            self.cbr2d_6=CBR2D(base*4,base,up=True)
            self.cbr2d_7=CBR2D(base*2,3,up=True)

    def foreground(self,x):
        h1=self.cbr3d_0(h)
        h2=self.cbr3d_1(h)
        h3=self.cbr3d_2(h)
        h4=self.cbr3d_3(h)
        h=self.cbr3d_4(F.concat([h,h]))
        h=self.cbr3d_5(F.concat([h3,h]))
        hm=self.cbr3d_6(F.concat([h2,h]))
        h=self.cbr3d_7(F.concat([h1,hm]))
        mask=F.dropout(F.sigmoid(self.c_m(hm)),ratio=0.5)

        return F.tanh(h),mask

    def background(self,x):
        h1=self.cbr2d_0(h)
        h2=self.cbr2d_1(h)
        h3=self.cbr2d_2(h)
        h4=self.cbr2d_3(h)
        h=self.cbr2d_4(F.concat([h,h]))
        h=self.cbr2d_5(F.concat([h3,h]))
        h=self.cbr2d_6(F.concat([h2,h]))
        h=self.cbr2d_7(F.concat([h1,h]))

        return F.tanh(h)

    def __call__(self,x):
        b,_,_,h,w=x.shape

        fg,mask=self.foreground(x)
        mask=F.tile(mask,(1,3,1,h,w))

        bg=self.background(x[:,:,0,:,:].reshape(b,3,w,h))
        bg=bg.reshape(b,3,1,h,w)
        bg=F.tile(bg,(1,1,64,1,1))

        return mask*fg + (1-mask)*bg

class Discriminator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.cbr3d_0=CBR3D(3,base,down=True)
            self.cbr3d_1=CBR3D(base,base*2,down=True)
            self.cbr3d_2=CBR3D(base*2,base*4,down=True)
            self.cbr3d_3=CBR3D(base*4,base*8,down=True)
            self.cbr3d_4=L.Linear(None,1,initialW=w)

    def __call__(self,x):
        h=self.cbr3d_0(x)
        h=self.cbr3d_1(h)
        h=self.cbr3d_2(h)
        h=self.cbr3d_3(h)
        h=self.cbr3d_4(h)

        return h
