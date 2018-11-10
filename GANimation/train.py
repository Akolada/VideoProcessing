import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import Discriminator,Predictor
from prepare import prepare_dataset

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description="GANimation")
parser.add_argument("--epochs",default=1000,type=int,help="the numbef of epochs")
parser.add_argument("--batchsize",default=20,type=int,help="batch size")
parser.add_argument("--testsize",default=2,type=int,help="test size")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iterations")
parser.add_argument("--Ntrain",default=1000,type=int,help="the number of train images")
parser.add_argument("--size",default=128,type=int,help="image width")

args=parser.parse_args()
epochs=args.epochs
batchsize=args.batchsize
testsize=args.testsize
iterations=args.iterations
Ntrain=args.Ntrain
size=args.size

outdir="./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

tenka_path="./tenka/"

discriminator=Discriminator()
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
        rnd=np.random.randint(1,Ntrain-batchsize)
        for index in range(batchsize):
            filename=tenka_path+"tenka_"+str(rnd+index)+".png"
            img=prepare_dataset(filename)
            frame_box.append(img)

        frame=chainer.as_variable(xp.array(frame_box).astype(xp.float32))
        generate=chainer.as_variable(xp.zeros_like(frame[0]).astype(xp.float32)).reshape(1,3,size,size)

        for time in range(batchsize-1):
            if time==0:
                blank=F.concat([generate,generate]).reshape(1,6,size,size)
                x=F.concat([frame[time].reshape(1,3,size,size),blank]).reshape(1,9,size,size)

            elif time==1:
                blank=chainer.as_variable(xp.zeros_like(frame[time]).astype(xp.float32)).reshape(1,3,size,size)
                x=F.concat([frame[time].reshape(1,3,size,size),frame[time-1].reshape(1,3,size,size)]).reshape(1,6,size,size)
                x=F.concat([x,blank]).reshape(1,9,size,size)

            else:
                x=F.concat([frame[time].reshape(1,3,size,size),frame[time-1].reshape(1,3,size,size)]).reshape(1,6,size,size)
                x=F.concat([x,frame[time-2].reshape(1,3,size,size)]).reshape(1,9,size,size)

            x_next=predictor(x)
            real=F.concat([frame[time+1].reshape(1,3,size,size),frame[time].reshape(1,3,size,size)]).reshape(1,6,size,size)
            fake=F.concat([x_next,generate]).reshape(1,6,size,size)

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
            fake=F.concat([x_next,generate]).reshape(1,6,size,size)
            dis_fake=discriminator(fake)

            gen_loss=F.mean(F.softplus(-dis_fake))

            predictor.cleargrads()
            gen_loss.backward()
            pred_opt.update()
            gen_loss.unchain_backward()

            sum_pred_loss+=gen_loss.data.get()
            sum_dis_loss+=dis_loss.data.get()

            generate=x_next

            if batch==0:
                serializers.save_npz("predictor.model",predictor)

    print("epoch:{}".format(epoch))
    print("Predictor loss:{}".format(sum_pred_loss/iterations))
    print("Discriminator loss:{}".format(sum_pred_loss/iterations))