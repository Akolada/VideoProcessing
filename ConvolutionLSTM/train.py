import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,optimizers,serializers
import numpy as np
import argparse
import os
import pylab
import cv2 as cv
from model import Discriminator,EncDec
from target  import target

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

def prepare_dataset(filename):
    image=cv.imread(filename)
    if image is not None:
        image = cv.resize(image,(32,32),interpolation=cv.INTER_CUBIC)
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
parser.add_argument("--iterations",default=10000,type=int,help="the number of iterations")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--weight",default=10.0,type=float,help="the weight of grad loss")

args=parser.parse_args()
framesize=args.framesize
epochs=args.epoch
iterations=args.iterations
interval=args.interval
weight=args.weight

image_path="/usr/MachineLearning/Dataset/cinderella/"
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
left=frames1[0].reshape(1,3,32,32)
right=frames2[0].reshape(1,3,32,32)

outdir="./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

encdec=EncDec()
encdec.to_gpu()
ed_opt=set_optimizer(encdec)

#discriminator=Discriminator()
#discriminator.to_gpu()
#dis_opt=set_optimizer(discriminator)

target = target()

for epoch in range(epochs):
    #sum_dis_loss=0
    sum_gen_loss=0
    for batch in range(0,iterations,framesize):
        frame_box=[]
        rnd=np.random.randint(list_len)
        dir_path=image_path+image_list[rnd]
        ta = np.random.choice(["lefteye","righteye"])
        for index in range(framesize):
            filename=dir_path + "/" + ta + "_" + str( index)+".png"
            frame=prepare_dataset(filename)
            frame_box.append(frame)

        frames=chainer.as_variable(xp.array(frame_box).astype(xp.float32))

        x=frames[0:framesize-1]
        t=frames[1:framesize]

        y=encdec(x)
        #y_dis=discriminator(y)
        #t_dis=discriminator(tar)

        #y.unchain_backward()

        #dis_loss=F.mean(F.softplus(y_dis))+F.mean(F.softplus(-t_dis))

        #discriminator.cleargrads()
        #dis_loss.backward()
        #dis_opt.update()
        #dis_loss.unchain_backward()

        #y=encdec(t)
        #y_dis=discriminator(y)

        #gen_loss=F.mean(F.softplus(-y_dis))
        gen_loss=F.mean_absolute_error(y,t)

        encdec.cleargrads()
        gen_loss.backward()
        ed_opt.update()
        gen_loss.unchain_backward()

        #sum_dis_loss+=dis_loss.data.get()
        sum_gen_loss+=gen_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("encdec.model",encdec)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            for i_ in range(framesize-1):
                with chainer.using_config("train",False):
                    left=encdec(left)
                    right = encdec(right)
                    yleft=left.data.get()
                    yright=right.data.get()
                    tmp_left=np.clip(yleft[0]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                    tmp_right=np.clip(yright[0]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                    tmp_left = cv.resize(tmp_left,(30,30),interpolation=cv.INTER_CUBIC)
                    tmp_right = cv.resize(tmp_right,(30,30),interpolation=cv.INTER_CUBIC)
                    test[leftlist[0]:leftlist[1], leftlist[2]:leftlist[3]] = tmp_left
                    test[rightlist[0]:rightlist[1], rightlist[2]:rightlist[3]] = tmp_right
                    pylab.subplot(4,4,i_+1)
                    pylab.imshow(test)
                    pylab.axis('off')
                    pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch:{}".format(epoch))
    #print("Discriminator loss:{}".format(sum_dis_loss/iterations/framesize))
    print("EncDec loss:{}".format(sum_gen_loss/iterations/framesize))