import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import argparse
import os
import pylab
from model import Generator,Discriminator

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha,beta):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

generator=Generator()
generator.to_gpu()
gen_opt=set_optimizer(generator)

discriminator=Discriminator()
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,iterations,batchsize):
        frame_box=[]
        for frame in range(framesize):
            

        frames=chainer.as_variable(xp.array(frame_box).astype(xp.float32))

        next=generator(frames)
        dis_x=discriminator(frames)
        dis_t=discriminator(target)

        dis_loss=F.mean(F.softplus(-dis_t))+F.mean(F.softplus(dis_x))

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        next=generator(frames)
        dis_x=discriminator(frmaes)

        gen_loss=F.mean(F.softplus(-dis_x))

        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss+=dis_loss.data.get()
        sum_gen_loss+=gen_loss.data.get()

        if batch==0:
            serializers.save_npz("generator.model",generator)

    print("epoch:{}".format(epoch))
    print("Generator loss:{}".format(sum_gen_loss/iterations))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))
