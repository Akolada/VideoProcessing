import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import Generator,Discriminator,UNet

xp=cuda.xupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description="RecycleGAN")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize",default=10,type=int,help="batchsize")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--testsize",default=2,type=int,help="testsize")
parser.add_argument("--weight",default=10.0,type=float,help="the weight of loss")

args=parser.parse_args()
epochs=args.epochs
batchsize=args.batchsize
interval=args.interval
testsize=args.testsize
weight=args.weight

generator_xy=Generator()
generator_xy.to_gpu()
gen_opt_xy=set_optimizer(generator_xy)

generator_yx=Generator()
generator_yx.to_gpu()
gen_opt_yx=set_optimizer(generator_yx)

discriminator_xy=Discriminator()
discriminator_xy.to_gpu()
dis_opt_xy=set_optimizer(discriminator_xy)

discriminator_yx=Discriminator()
discriminator_yx.to_gpu()
dis_opt_yx=set_optimizer(discriminator_yx)

predictor_x=UNet()
predictor_x.to_gpu()
pre_opt_x=set_optimizer(predictor_x)

predictor_y=UNet()
predictor_y.to_gpu()
pre_opt_y=set_optimizer(predictor_y)

for epoch in range(epochs):
    sum_dis_loss=0
    sum_gen_loss=0
    for batch in range(0,Ntrain,batchsize):
        x_box=[]
        y_box=[]
        for index in range(batchsize):
            # Incomplete
            
        x=chainer.as_variable(xp.array(x_box).astype(xp.float32))
        y=chainer.as_variable(xp.array(y_box).astype(xp.float32))
        
        x_series_box=[]
        y_serise_box=[]
        for index in range(tuple, time):
            x_sum=x[index:index+2]
            y_sum=y[index:index+2]
            x_series=F.concat([x[index-1],x[index-2]])
            y_series=F.concat([y[index-1],y[index-2]])
            
            x_y=generator_xy(x_sum)
            x_y_x=generator_yx(x_y)

            y_x=generator_yx(y_sum)
            y_x_y=generator_xy(y_x)

            fake_xy=discriminator_xy(x_y)
            real_xy=discriminator_xy(y_sum)
            fake_yx=discriminator_yx(y_x)
            real_yx=discriminator_yx(x_sum)

            dis_loss_y=F.mean(F.softplus(fake_xy))+F.mean(F.softplus(-real_xy))
            dis_loss_x=F.mean(F.softplus(fake_yx))+F.mean(F.softplus(-real_yx))

            discriminator_xy.cleargrads()
            dis_loss_y.backward()
            dis_opt_xy.update()
            dis_loss_y.unchain_backward()

            discriminator_yx.cleargrads()
            dis_loss_yx.backward()
            dis_opt_yx.update()
            dis_loss_yx.unchain_backward()

            gen_loss_xy=F.mean(F.softplus(-fake_xy))
            gen_loss_yx=F.mean(F.softplus(-fake_yx))

            cycle_loss_x=F.mean_squared_error(x_y_x,x_sum)
            cycle_loss_y=F.mean_squared_error(y_x_y,y_sum)

            x_next=predictor_x(x_series)
            recurrent_loss_x=F.mean_squared_error(x_next,x_sum[2])
            x_next=generator_yx(predictor_y(generator_xy(x_series)))
            recycle_loss_x=F.mean_squared_error(x_next,x_sum[2])

            y_next=predictor_y(y_series)
            recurrent_loss_y=F.mean_squared_error(y_next,y_sum[2])
            y_next=generator_xy(predictor_x(generator_yx(y_series)))
            recycle_loss_y=F.mean_squared_error(y_next,y_sum[2])

            gen_loss_x=gen_loss_xy+cycle_loss_x+weight*(recurrent_loss_x+recycle_loss_x)
            gen_loss_y=gen_loss_yx+cycle_loss_y+weight*(recurrent_loss_y+recycle_loss_y)

            generator_xy.cleargrads()
            predictor_x.cleargrads()
            gen_loss_x.backward()
            gen_opt_xy.update()
            pre_opt_x.update()

            generator_yx.cleargrads()
            predictor_y.cleargrads()
            gen_loss_y.backward()
            gen_opt_yx.update()
            pre_opt_y.update()

        sum_gen_loss+=(gen_loss_x+gen_loss_y)
        sum_dis_loss+=(dis_loss_xy+dis_loss_yx)

        if epoch%interval==0 and batch==0:
            serializers.save_npz("generator_xy.model",generator_xy)
            serializers.save_npz("generator_yx.model",generator_yx)

    print("epoch:{}".format(epoch))
    print("Generator loss:{}".format(sum_gen_loss/Ntrain))
    print("Discriminator loss:{}".format(sum_dis_loss/Ntrain))
