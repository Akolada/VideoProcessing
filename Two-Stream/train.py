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

def prepare_test(filename):
    image = cv.imread(filename)
    if image is not None:
        image = cv.resize(image,(128,128),interpolation=cv.INTER_CUBIC)
        height,width = image.shape[0],image.shape[1]

        leftdown = int(height/2)-20
        leftup = int(height/2)+10
        rightdown = int(width/2)-35
        rightup = int(width/2)-5
        lefteye = image[leftdown:leftup, rightdown:rightup]
        lefteye = cv.resize(lefteye,(32,32),interpolation=cv.INTER_CUBIC)
        lefteye = lefteye[:,:,::-1].transpose(2,0,1)
        lefteye = (lefteye - 127.5) / 127.5
        leftlist = [leftdown, leftup, rightdown, rightup]

        leftdown = int(height/2)-20
        leftup = int(height/2)+10
        rightdown = int(width/2)+5
        rightup = int(width/2)+35
        righteye = image[leftdown:leftup, rightdown:rightup]
        righteye = cv.resize(righteye,(32,32),interpolation=cv.INTER_CUBIC)
        righteye = righteye[:,:,::-1].transpose(2,0,1)
        righteye = (righteye - 127.5) / 127.5
        rightlist = [leftdown, leftup, rightdown, rightup]

        image = image[:,:,::-1]
        return image,lefteye, leftlist, righteye, rightlist

parser=argparse.ArgumentParser(description="Two-Stream")
parser.add_argument("--framesize",default=16,type=int,help="frame size")
parser.add_argument("--epoch",default=1000,type=int,help="the number of epochs")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iterations")
parser.add_argument("--interval",default=5,type=int,help="the interval of snapshot")
parser.add_argument("--weight",default=5.0,type=float,help="the weight of grad loss")
parser.add_argument("--ndis",default=1,type=int,help="the number of discriminator update")
parser.add_argument("--size",default=32,type=int,help="the image width")

args=parser.parse_args()
framesize=args.framesize
epochs=args.epoch
iterations=args.iterations
interval=args.interval
weight=args.weight
n_dis=args.ndis
size=args.size

outdir="./output"
if not os.path.exists(outdir):
    os.mkdir(outdir)

image_path="/image/"
image_list=os.listdir(image_path)
list_len=len(image_list)

lefteye_box = []
righteye_box = []
image_name="./test.png"
test, lefteye, leftlist, righteye, rightlist =prepare_test(image_name)
lefteye_box.append(lefteye)
righteye_box.append(righteye)
frames1=chainer.as_variable(xp.array(lefteye_box).astype(xp.float32))
frames2=chainer.as_variable(xp.array(righteye_box).astype(xp.float32))
x1=frames1[0].reshape(1,3,size,size)
x2=frames2[0].reshape(1,3,size,size)
xtest=F.concat([x1,x2],axis=0)

generator=Generator()
generator.to_gpu()
gen_opt=set_optimizer(generator)

discriminator=Discriminator()
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,iterations,framesize):
        for _ in range(n_dis):
            lefteye_box = []
            righteye_box = []
            rnd=np.random.randint(list_len)
            dir_path=image_path+image_list[rnd]
            for index in range(framesize):
                filename=dir_path + "/lefteye_" + str(index) + ".png"
                frame=prepare_dataset(filename)
                lefteye_box.append(frame)
                filename=dir_path+"/righteye_"+str(index)+".png"
                frame=prepare_dataset(filename)
                righteye_box.append(frame)

            frames1=chainer.as_variable(xp.array(lefteye_box).astype(xp.float32))
            frames2=chainer.as_variable(xp.array(righteye_box).astype(xp.float32))
            x1=frames1[0].reshape(1,3,size,size)
            x2=frames2[0].reshape(1,3,size,size)
            x=F.concat([x1,x2],axis=0)
            target1=frames1.transpose(1,0,2,3).reshape(1,3,framesize,size,size)
            target2=frames2.transpose(1,0,2,3).reshape(1,3,framesize,size,size)
            target=F.concat([target1,target2],axis=0)
            target=F.concat([target, target],axis=2)

            y=generator(x)
            y=F.concat([y,y],axis=2)
            dis_x=discriminator(y)
            dis_t=discriminator(target)

            y.unchain_backward()

            dis_loss=F.mean(F.softplus(-dis_t))+F.mean(F.softplus(dis_x))

            discriminator.cleargrads()
            dis_loss.backward()
            dis_opt.update()
            dis_loss.unchain_backward()

        y=generator(x)
        y=F.concat([y,y],axis=2)
        dis_x=discriminator(y)

        gen_loss=F.mean(F.softplus(-dis_x))
        gen_loss+=weight*F.mean_absolute_error(y,target)

        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss+=dis_loss.data.get()
        sum_gen_loss+=gen_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("generator.model",generator)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            with chainer.using_config("train",False):
                y=generator(xtest)
            y=y.data.get()
            yleft=y[0].transpose(1,0,2,3)
            yright=y[1].transpose(1,0,2,3)
            for i_ in range(framesize):
                tmp_left = (np.clip(yleft[i_]*127.5+127.5,0,255)).transpose(1,2,0).astype(np.uint8)
                tmp_right = (np.clip(yright[i_]*127.5+127.5,0,255)).transpose(1,2,0).astype(np.uint8)
                tmp_left = cv.resize(tmp_left,(30,30),interpolation=cv.INTER_CUBIC)
                tmp_right = cv.resize(tmp_right,(30,30),interpolation=cv.INTER_CUBIC)
                test[leftlist[0]:leftlist[1], leftlist[2]:leftlist[3]] = tmp_left
                test[rightlist[0]:rightlist[1], rightlist[2]:rightlist[3]] = tmp_right
                pylab.subplot(4,4,i_+1)
                pylab.imshow(test)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch:{}".format(epoch))
    print("Generator loss:{}".format(sum_gen_loss/iterations))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))
