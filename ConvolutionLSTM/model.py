import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers,Variable
from instance_normalization import InstanceNormalization

class CBR(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False):
        w=initializers.Normal(0.02)
        self.up=up
        self.down=down
        super(CBR,self).__init__()
        with self.init_scope():
            self.cpara=L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.Convolution2D(in_ch,out_ch,4,2,1,initialW=w)

            self.bn0=InstanceNormalization(out_ch)

    def __call__(self,x):
        if self.up:
            h=F.unpooling_2d(x,2,2,0,cover_all=False)
            h=F.relu(self.bn0(self.cpara(h)))

        elif self.down:
            h=F.relu(self.bn0(self.cdown(x)))

        else:
            h=F.relu(self.bn0(self.cpara(x)))

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
            self.cbr0=CBR(3,base,down=True)
            self.cbr1=CBR(base,base*2,down=True)
            self.cbr2=CBR(base*2,base*4,down=True)
            self.clstm0 = ConvLSTM(base*4,base*4,3)
            self.cbr3 = CBR(base*4,base*2,up=True)
            self.cbr4 = CBR(base*2,base,up=True)
            self.cbr5 = CBR(base,base,up=True)
            self.cbr6 = L.Convolution2D(base,3,3,1,1,initialW=w)

    def __call__(self,x):
        self.clstm0.reset_state()
        
        b=x.shape[0]
        h=self.cbr0(x)
        h=self.cbr1(h)
        h=self.cbr2(h)
        h=self.clstm0(h)
        h=self.cbr3(h)
        h=self.cbr4(h)
        h=self.cbr5(h)
        h=self.cbr6(h)

        return h

class Encoder(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(Encoder,self).__init__()
        with self.init_scope():
            self.cbr0 = CBR(3, base, down=True)
            self.cbr1 = CBR(base, base*2, down=True)
            self.cbr2 = CBR(base*2, base*4, down=True)
            self.cbr3 = CBR(base*4, base*8, down=True)
            self.l0 = L.Linear(None,512)
            self.lstm0 = L.LSTM(base*8, base*8)
            self.lstm1 = L.LSTM(base*8, base*8)
            self.l1 = L.Linear(512,8)

    def __call__(self,x):
        self.lstm0.reset_state()
        self.lstm1.reset_state()
        
        h=self.cbr0(x)
        h=self.cbr1(h)
        h=self.cbr2(h)
        h=self.cbr3(h)
        h=F.relu(self.l0(h))
        h=F.relu(self.lstm0(h))
        h=F.relu(self.lstm1(h))
        h=self.l1(h)

        return h

class Discriminator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.down0=CBR(3,base,down=True)
            self.down1=CBR(base,base*2,down=True)
            self.down2=CBR(base*2,base*4,down=True)
            self.down3=CBR(base*4,base*8,down=True)
            self.lout=L.Linear(None,1,initialW=w)

    def __call__(self,x):
        h=self.down0(x)
        h=self.down1(h)
        h=self.down2(h)
        h=self.down3(h)
        h=self.lout(h)

        return h