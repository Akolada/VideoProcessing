import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,optimizers,serializers
import numpy as np
import argparse
import os
import pylab
from model import Discriminator,EncDec

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

outdir="./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

encdec=EncDec()
encdec.to_gpu()
ed_opt=set_optimizer(encdec)

discriminator=Discriminator()
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

for epoch in range(epochs):
    sum_dis_loss=0
    sum_gen_loss=0
    for batch in range(0,iterations,batchsize):
        frame_box=[]
        for index in range(batchsize):


        frame=xp.array(frame_box).astype(xp.float32)
        frame=chainer.as_variable(frame)

        x=frame[0:batchsize-1]
        t=frame[1:batchsize]

        y=encdec(x)
        y_dis=discriminator(y)
        t_dis=discriminator(t)

        y.unchain_backward()

        dis_loss=F.mean(F.softplus(y_dis))+F.mean(F.softplus(-t_dis))

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        y=encdec(x)
        y_dis=discriminator(y)

        gen_loss=F.mean(F.softplus(-y_dis))
        gen_loss+=weight*F.mean_absolute_error(y,t)

        sum_dis_loss+=dis_loss.data.get()
        sum_gen_loss+=gen_loss.data.get()

        if batch==0:
            serializers.save_npz("encdec.model",encdec)

    print("epoch:{}".format(epoch))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))
    print("EncDec loss:{}".format(sum_gen_loss/iterations))
