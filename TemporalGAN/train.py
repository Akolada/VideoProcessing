import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
import cv2 as cv
from model import ImageGenerator, TemporalGenerator, Discriminator

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002, beta=0.5):
    optimizer = optimizers.Adam(alpha=alpha,beta1 = beta)
    optimizer.setup(model)

    return optimizer

def calc_loss(real,fake):
    _,c,w,h = real.shape

    return F.mean_absolute_error(real,fake) / (c*w*h)

def prepare_dataset(filename):
    image=cv.imread(filename)
    if image is not None:
        image = cv.resize(image,(size,size),interpolation=cv.INTER_CUBIC)
        hr_image = image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        return hr_image

parser = argparse.ArgumentParser(description="TemporalGAN")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--framesize",default=16,type=int,help="frame size")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iterations")
parser.add_argument("--weight",default=10.0,type=float,help="the weight of content loss")
parser.add_argument("--size",default=64,type=int,help="image width")
parser.add_argument("--batchsize", default=1, type=int, help="batch size")

args=parser.parse_args()
epochs = args.epochs
framesize = args.framesize
iterations = args.iterations
weight = args.weight
size = args.size
batchsize = args.batchsize

image_path = "./syani"
image_list = os.listdir(image_path)
image_len = len(image_list)

outdir = "./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

i_generator = ImageGenerator()
i_generator.to_gpu()
ig_opt = set_optimizer(i_generator)

t_generator = TemporalGenerator()
t_generator.to_gpu()
tg_opt = set_optimizer(t_generator)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt  = set_optimizer(discriminator)

ztest = xp.random.uniform(-1,1,(batchsize * framesize, 128)).astype(xp.float32)
ztest = chainer.as_variable(ztest)

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
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
        z = xp.random.uniform(-1,1,(batchsize * framesize, 128)).astype(xp.float32)
        z = chainer.as_variable(z)

        tz = t_generator(z)
        y = i_generator(z,tz)
        _, channels, height, width = y.shape
        t_3d = t.reshape(batchsize, framesize, channels, height, width).transpose(0,2,1,3,4)
        y_3d = y.reshape(batchsize, framesize, channels, height, width).transpose(0,2,1,3,4)
        t_dis = discriminator(t_3d)
        y_dis = discriminator(y_3d)
        dis_loss = F.mean(F.softplus(-t_dis)) + F.mean(F.softplus(y_dis))

        y.unchain_backward()

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        z = xp.random.uniform(-1,1,(batchsize * framesize, 128)).astype(xp.float32)
        z = chainer.as_variable(z)

        tz = t_generator(z)
        y = i_generator(z,tz)
        y_3d = y.reshape(batchsize,framesize,channels,height,width).transpose(0,2,1,3,4)
        y_dis = discriminator(y_3d)
        gen_loss = F.mean(F.softplus(-y_dis))
        #gen_loss += weight * calc_loss(y,t)

        i_generator.cleargrads()
        t_generator.cleargrads()
        gen_loss.backward()
        ig_opt.update()
        tg_opt.update()

        gen_loss.unchain_backward()

        for p in discriminator.params():
            p.data = xp.clip(p.data,-0.01,0.01)

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if batch == 0:
            serializers.save_npz("image_generator.model",i_generator)
            serializers.save_npz("temporal_generator.model",t_generator)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            with chainer.using_config("train",False):
                tz = t_generator(ztest)
                y = i_generator(z,tz)
            y = y[ : 16].data.get()
            for i in range(framesize):
                tmp = np.clip(y[i]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(4,4,i+1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch : {}".format(epoch))
    print("Discriminator Loss : {}".format(sum_dis_loss/iterations))
    print("Generator loss : {}".format(sum_gen_loss/iterations))