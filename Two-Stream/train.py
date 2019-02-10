import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import argparse
import os
import pylab
import cv2 as cv
from model import Generator,Discriminator,VGG

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

def prepare_dataset(filename):
    image=cv.imread(filename)
    if image is not None:
        image = cv.resize(image,(size,size),interpolation=cv.INTER_CUBIC)
        hr_image = image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        return hr_image

parser=argparse.ArgumentParser(description="Two-Stream")
parser.add_argument("--framesize",default=16,type=int,help="frame size")
parser.add_argument("--epoch",default=1000,type=int,help="the number of epochs")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iterations")
parser.add_argument("--interval",default=5,type=int,help="the interval of snapshot")
parser.add_argument("--weight",default=5.0,type=float,help="the weight of grad loss")
parser.add_argument("--ndis",default=2,type=int,help="the number of discriminator update")
parser.add_argument("--size",default=64,type=int,help="the image width")
parser.add_argument("--batchsize", default=2, type=int, help="batch size")

args=parser.parse_args()
framesize=args.framesize
epochs=args.epoch
iterations=args.iterations
interval=args.interval
weight=args.weight
n_dis=args.ndis
size=args.size
batchsize = args.batchsize

outdir="./output"
if not os.path.exists(outdir):
    os.mkdir(outdir)

image_path="./syani"
image_list=os.listdir(image_path)
list_len=len(image_list)

generator=Generator()
generator.to_gpu()
gen_opt=set_optimizer(generator)

discriminator=Discriminator()
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

ztest = xp.random.uniform(-1,1,(batchsize, 128)).astype(xp.float32)
ztest = chainer.as_variable(ztest)

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,iterations,framesize):
        batch_box = []
        for _ in range(batchsize):
            frame_box = []
            start_frame = np.random.randint(1, 1000 - framesize)
            for index in range(framesize):
                filename=image_path + "tenka_" + str(start_frame + index) + ".png"
                frame=prepare_dataset(filename)
                frame_box.append(frame)

            frames = xp.array(frame_box).astype(xp.float32).transpose(1,0,2,3).reshape(1,3,16,64,64)
            batch_box.append(frames)

        t = chainer.as_variable(xp.concatenate(batch_box))
        z = xp.random.uniform(-1,1,(batchsize, 128)).astype(xp.float32)
        z = chainer.as_variable(z)

        y=generator(z)
        dis_x=discriminator(y)
        dis_t=discriminator(t)

        y.unchain_backward()

        dis_loss=F.mean(F.softplus(-dis_t))+F.mean(F.softplus(dis_x))

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        z = xp.random.uniform(-1,1,(batchsize, 128)).astype(xp.float32)
        z = chainer.as_variable(z)

        y=generator(z)
        dis_x=discriminator(y)

        gen_loss=F.mean(F.softplus(-dis_x))

        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        gen_loss.unchain_backward()

        for p in discriminator.params():
            p.data = xp.clip(p.data,-0.01,0.01)

        sum_dis_loss+=dis_loss.data.get()
        sum_gen_loss+=gen_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("generator.model",generator)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            with chainer.using_config("train",False):
                y=generator(ztest)
            y=y.data.get()
            y=y[0].transpose(1,0,2,3)
            for i_ in range(framesize):
                tmp = np.clip(y[i_]*127.5 + 127.5, 0,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(4,4,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch:{}".format(epoch))
    print("Generator loss:{}".format(sum_gen_loss/iterations))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))
