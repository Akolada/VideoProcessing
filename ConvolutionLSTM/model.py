import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import cuda,Chain,initializers,Variable,serializers
from instance_normalization import InstanceNormalization

xp=cuda.cupy
cuda.get_device(0).use()

class VGG(Chain):
    def __init__(self):
        super(VGG,self).__init__()
        with self.init_scope():
            self.base = L.VGG16Layers()

    def __call__(self,x):
        h = self.base(x,layers=["conv5_3"])["conv5_3"]

        return h

class ConvLSTM(Chain):
    def __init__(self, inp = 256, mid = 128, sz = 5):
        super(ConvLSTM, self).__init__()
        with self.init_scope():
            self.wxi = L.Convolution2D(inp, mid, sz, pad = sz//2)
            self.whi = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True)
            self.wxf = L.Convolution2D(inp, mid, sz, pad = sz//2)
            self.whf = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True)
            self.wxc = L.Convolution2D(inp, mid, sz, pad = sz//2)
            self.whc = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True)
            self.wxo = L.Convolution2D(inp, mid, sz, pad = sz//2)
            self.who = L.Convolution2D(mid, mid, sz, pad = sz//2, nobias = True)

        self.inp = inp
        self.mid = mid

        self.pc = None
        self.ph = None

        with self.init_scope():
            Wci_initializer = initializers.Zero()
            self.Wci = chainer.variable.Parameter(Wci_initializer)
            Wcf_initializer = initializers.Zero()
            self.Wcf = chainer.variable.Parameter(Wcf_initializer)
            Wco_initializer = initializers.Zero()
            self.Wco = chainer.variable.Parameter(Wco_initializer)

    def reset_state(self, pc = None, ph = None):
        self.pc = pc
        self.ph = ph

    def initialize_params(self, shape):
        self.Wci.initialize((self.mid, shape[2], shape[3]))
        self.Wcf.initialize((self.mid, shape[2], shape[3]))
        self.Wco.initialize((self.mid, shape[2], shape[3]))

    def initialize_state(self, shape):
        self.pc = Variable(self.xp.zeros((shape[0], self.mid, shape[2], shape[3]), dtype = self.xp.float32))
        self.ph = Variable(self.xp.zeros((shape[0], self.mid, shape[2], shape[3]), dtype = self.xp.float32))
        
    def __call__(self, x):
        if self.Wci.data is None:
            self.initialize_params(x.data.shape)

        if self.pc is None:
            self.initialize_state(x.data.shape)

        ci = F.sigmoid(self.wxi(x) + self.whi(self.ph) + F.scale(self.pc, self.Wci, 1))
        cf = F.sigmoid(self.wxf(x) + self.whf(self.ph) + F.scale(self.pc, self.Wcf, 1))
        cc = cf * self.pc + ci * F.tanh(self.wxc(x) + self.whc(self.ph))
        co = F.sigmoid(self.wxo(x) + self.who(self.ph) + F.scale(cc, self.Wco, 1))
        ch = co * F.tanh(cc)

        self.pc = cc
        self.ph = ch
        
        return ch

class EncDec(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(EncDec,self).__init__()
        with self.init_scope():
            self.cbr0=CBR(3,base)
            #self.cbr1=CBR(base,base*2,down=True)
            #self.cbr2=CBR(base*2,base*4,down=True)
            #self.cbr3=CBR(base*4,base*8,down=True)
            #self.cbr4=CBR(base*8,base*16,down=True)
            self.clstm=ConvLSTM(base,base,3)
            #self.up0=CBR(base*16,base*8,up=True)
            #self.up1=CBR(base*8,base*4,up=True)
            #self.up2=CBR(base*4,base*2,up=True)
            #self.up3=CBR(base*2,base,up=True)
            self.up4=CBR(base,3)

    def __call__(self,x):
        self.clstm.reset_state()

        b=x.shape[0]
        h=self.cbr0(x)
        #h=self.cbr1(h)
        #h=self.cbr2(h)
        #h=self.cbr3(h)
        #h=self.cbr4(h)
        hout=self.clstm(h)
        #h=self.up0(h)
        #h=self.up1(h)
        #h=self.up2(h)
        #h=self.up3(h)
        h=self.up4(h)

        return hout

class FeatureExtractor(Chain):
    def __init__(self,base=64):
        super(FeatureExtractor,self).__init__()
        with self.init_scope():
            self.cbr0=CBR(3,base)
            self.clstm0=ConvLSTM(base,base,3)
            self.up4=CBR(base,3,up=True)
            
    def __call__(self,x):
        self.clstm0.reset_state()

        h = self.cbr0(x)
        hout = self.clstm0(h)
        h = self.up4(hout)
        
        return F.tanh(hout)

class FeatureEmbedding(Chain):
    def __init__(self):
        super(FeatureEmbedding,self).__init__()
        #w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.l0 = L.Linear(None,1024)
            self.lstm = L.LSTM(1024,128)
            self.l1 = L.Linear(128,8)   

    def __call__(self,x):
        h = self.l0(x)
        h = self.lstm(h)
        h = self.l1(h)

        return h

class Condition(Chain):
    def __init__(self):
        super(Condition,self).__init__()
        with self.init_scope():
            self.fextract = FeatureExtractor()
            self.fembed = FeatureEmbedding()

    def __call__(self,x,t):
        xfe = self.fextract(x)
        tfe = self.fextract(t)
        c = self.fembed(xfe-tfe)

        return xfe-tfe,c

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

            self.bn0=InstanceNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h=F.unpooling_2d(x,2,2,0,cover_all=False)
            h=self.activation(self.bn0(self.cpara(h)))

        elif self.down:
            h=self.activation(self.bn0(self.cdown(x)))

        else:
            h=self.activation(self.bn0(self.cpara(x)))

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
        super(Predictor,self).__init__()
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.c0=CBR(11,base)
            self.down0=CBR(base,base*2,down=True)
            self.down1=CBR(base*2,base*4,down=True)
            self.down2=CBR(base*4,base*8,down=True)
            #self.down3=CBR(base*8,base*16,down=True)
            #self.up0=CBR(base*32,base*8,up=True)
            self.up1=CBR(base*16,base*4,up=True)
            self.up2=CBR(base*8,base*2,up=True)
            self.up3=CBR(base*4,base,up=True)
            self.c1=L.Convolution2D(base*2,3,3,1,1,initialW=w)

    def __call__(self,x):
        h1=self.c0(x)
        h2=self.down0(h1)
        h3=self.down1(h2)
        h4=self.down2(h3)
        #h5=self.down3(h4)
        #h=self.up0(F.concat([h5,h5]))
        h=self.up1(F.concat([h4,h4]))
        h=self.up2(F.concat([h3,h]))
        h=self.up3(F.concat([h2,h]))
        h=self.c1(F.concat([h1,h]))

        return F.tanh(h)

class Discriminator_image(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(Discriminator_image,self).__init__()
        with self.init_scope():
            self.down0=CBR(3,base,down=True)
            self.down1=CBR(base,base*2,down=True)
            self.down2=CBR(base*2,base*4,down=True)
            self.down3=CBR(base*4,base*4,down=True)
            self.cout=L.Convolution2D(base*4,1,3,1,1,initialW=w)

    def __call__(self,x):
        h=self.down0(x)
        h=self.down1(h)
        h=self.down2(h)
        h=self.down3(h)
        h=self.cout(h)

        return h

class Discriminator_stream(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(Discriminator_stream,self).__init__()
        with self.init_scope():
            self.down0=CBR(8,base,down=True)
            self.down1=CBR(base,base*2,down=True)
            self.down2=CBR(base*2,base*4,down=True)
            self.down3=CBR(base*4,base*4,down=True)
            self.cout=L.Convolution2D(base*4,1,3,1,1,initialW=w)

    def __call__(self,x):
        h=self.down0(x)
        h=self.down1(h)
        h=self.down2(h)
        h=self.down3(h)
        h=self.cout(h)

        return h