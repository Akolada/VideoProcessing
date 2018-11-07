import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import Discriminator,Predictor

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha,beta):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description="GANimation")
parser.add_argument("--epochs",default=1000,type=int,help="the numbef of epochs")
parser.add_argument("--batchsize",default=20,type=int,help="batch size")
parser.add_argument("--testsize",default=2,type=int,help="test size")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iterations")

args=parser.parse_args()
epochs=args.epochs
batchsize=args.batchsize
testsize=args.testsize
iterations=args.iterations

outdir="./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

discriminator=Discriminator
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

predictor=Predictor()
predictor.to_gpu()
pred_opt=set_optimizer(predictor)

for epoch in range(epochs):
    sum_dis_loss=0
    sum_pred_loss=0
    for batch in range(0,iterations,batchsize):
        frame_box=[]
        for index in range(batchsize):


        frame=chainer.as_variable(xp.array(frame_box).astype(xp.flost32))

        for time in range(batchsize-1):
            if time==0:
                blank=chainer.as_variable(xp.zeros_like(frame[time]).astype(xp.float32))
                blank=F.concat([blank,blank])
                x=F.concat([frame[time],blank])

            elif time==1:
                blank=chainer.as_variable(xp.zeros_like(frame[time]).astype(xp.float32))
                x=F.concat(generate,frame[time-1]])
                x=F.concat([x,blank])

            else:
                x=F.concat([generate,frame[time-1]])
                x=F.concat([x,frame[time-2]])

            x_next=predictor(x)
            real=F.concat([frame[time+1],frame[time]])
            fake=F.concat([x_next,generate])
            dis_real=discriminator(real)
            dis_fake=discriminator(fake)

            dis_loss=F.mean(F.softplus(-dis_real))
            dis_loss+=F.mean(F.softplus(dis_fake))

            x_next.unchain_backward()

            discriminator.cleargrads()
            dis_loss.backward()
            dis_opt.update()
            dis_loss.unchain_backward()
            
            x_next=predictor(x)
            fake=F.concat([x_next,generate])
            dis_fake=discriminator(fake)

            gen_loss=F.mean(F.softplus(-dis_fake))

            predictor.cleargrads()
            gen_loss.backward()
            pred_opt.update()
            gen_loss.unchain_backward()

            sum_pred_loss+=gen_loss.data.get()
            sum_dis_loss+=dis_loss.data.get()

            if batch==0:
                serializers.save_npz("predictor.model",predictor)

    print("epoch:{}".format(epoch))
    print("Predictor loss:{}".format(sum_pred_loss/iteratons))
    print("Discriminator loss:{}".format(sum_pred_loss/iterations))
