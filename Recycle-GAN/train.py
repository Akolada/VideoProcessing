import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import Generator,Discriminator,UNet
from prepare import prepare_dataset

xp=cuda.cupy
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
parser.add_argument("--frames",default=3,type=int,help="the number of frames")
parser.add_argument("--times",default=20,type=int,help="times")
parser.add_argument("--iterations",default=2000,type=int,help="the numbef of iterations")
parser.add_argument("--size",default=128,type=int,help="the size of images")

args=parser.parse_args()
epochs=args.epochs
batchsize=args.batchsize
interval=args.interval
testsize=args.testsize
weight=args.weight
frames=args.frames
times = args.times
iterations=args.iterations
size = args.size

model_dir="./model/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

x_path="/x/"
y_path="/y/"
x_len=len(os.listdir(x_path))
y_len=len(os.listdir(y_path))

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
    for batch in range(0,iterations,batchsize):
        x_box=[]
        y_box=[]
        rnd1 = np.random.randint(x_len - batchsize)
        rnd2 = np.random.randint(y_len - batchsize)
        for index in range(batchsize):
            image_name = x_path + str(rnd1+index)+".png"
            source = prepare_dataset(image_name)
            image_name = y_path + str(rnd2+index)+".png"
            target = prepare_dataset(image_name)
            x_box.append(source)
            y_box.append(target)
            
        x=chainer.as_variable(xp.array(x_box).astype(xp.float32))
        y=chainer.as_variable(xp.array(y_box).astype(xp.float32))
        
        for index in range(frames, batchsize):
            x_series=F.concat([x[index-2].reshape(1,3,size,size),x[index-1].reshape(1,3,size,size)])
            x_serial = x[index-2:index]
            y_series=F.concat([y[index-2].reshape(1,3,size,size),y[index-1].reshape(1,3,size,size)])
            y_serial = y[index-2:index]
            
            x_y=generator_xy(x_serial)
            y_x=generator_yx(y_serial)

            fake_xy=discriminator_xy(x_y)
            real_xy=discriminator_xy(y_serial)
            fake_yx=discriminator_yx(y_x)
            real_yx=discriminator_yx(x_serial)

            x_y.unchain_backward()
            y_x.unchain_backward()

            dis_loss_y=F.mean(F.softplus(fake_xy))+F.mean(F.softplus(-real_xy))
            dis_loss_x=F.mean(F.softplus(fake_yx))+F.mean(F.softplus(-real_yx))

            discriminator_xy.cleargrads()
            dis_loss_y.backward()
            dis_opt_xy.update()
            dis_loss_y.unchain_backward()

            discriminator_yx.cleargrads()
            dis_loss_x.backward()
            dis_opt_yx.update()
            dis_loss_x.unchain_backward()

            x_y=generator_xy(x_serial)
            x_y_x=generator_yx(x_y)
            y_x=generator_yx(y_serial)
            y_x_y=generator_xy(y_x)

            fake_xy=discriminator_xy(x_y)
            fake_yx=discriminator_yx(y_x)

            gen_loss_xy=F.mean(F.softplus(-fake_xy))
            gen_loss_yx=F.mean(F.softplus(-fake_yx))

            cycle_loss_x=F.mean_squared_error(x_y_x,x_serial)
            cycle_loss_y=F.mean_squared_error(y_x_y,y_serial)

            x_next=predictor_x(x_series)
            recurrent_loss_x=F.mean_squared_error(x_next,x[index].reshape(1,3,size,size))

            x_y_serial = (generator_xy(x_serial))
            x_y_series = F.concat([x_y_serial[0].reshape(1,3,size,size),x_y_serial[1].reshape(1,3,size,size)])
            x_next = generator_yx(predictor_y(x_y_series))
            recycle_loss_x=F.mean_squared_error(x_next,x[index].reshape(1,3,size,size))

            y_next=predictor_y(y_series)
            recurrent_loss_y=F.mean_squared_error(y_next,y[index].reshape(1,3,size,size))

            y_x_serial = (generator_yx(y_serial))
            y_x_series = F.concat([y_x_serial[0].reshape(1,3,size,size),y_x_serial[1].reshape(1,3,size,size)])
            y_next = generator_xy(predictor_x(y_x_series))
            recycle_loss_y=F.mean_squared_error(y_next,y[index].reshape(1,3,size,size))

            gen_loss_x=gen_loss_xy+weight*(cycle_loss_x + recurrent_loss_x + recycle_loss_x)
            gen_loss_y=gen_loss_yx+weight*(cycle_loss_y + recurrent_loss_y + recycle_loss_y)
            gen_loss=gen_loss_x + gen_loss_y
            
            generator_xy.cleargrads()
            generator_yx.cleargrads()
            predictor_x.cleargrads()
            predictor_y.cleargrads()

            gen_loss.backward()

            gen_opt_xy.update()
            pre_opt_x.update()
            gen_opt_yx.update()
            pre_opt_y.update()
            
            gen_loss.unchain_backward()

        sum_gen_loss+=(gen_loss)
        sum_dis_loss+=(dis_loss_y+dis_loss_x)

        if epoch%interval==0 and batch==0:
            serializers.save_npz(model_dir + "generator_xy_{}.model".format(epoch),generator_xy)
            serializers.save_npz(model_dir + "generator_yx_{}.model".format(epoch),generator_yx)
            serializers.save_npz(model_dir + "predictor_y_{}".format(epoch),predictor_y)

    print("epoch:{}".format(epoch))
    print("Generator loss:{}".format(sum_gen_loss/iterations))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))