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

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

def prepare_dataset(filename):
    image=cv.imread(filename)
    if image is not None:
        image = cv.resize(image,(128,128),interpolation=cv.INTER_CUBIC)
        hr_image = image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        return hr_image

parser=argparse.ArgumentParser(description="Two-Stream")
parser.add_argument("--framesize",default=16,type=int,help="frame size")
parser.add_argument("--epoch",default=1000,type=int,help="the number of epochs")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iterations")
parser.add_argument("--interval",default=5,type=int,help="the interval of snapshot")
parser.add_argument("--weight",default=10.0,type=float,help="the weight of grad loss")

args=parser.parse_args()
framesize=args.framesize
epochs=args.epoch
iterations=args.iterations
interval=args.interval
weight=args.weight

image_path="./image/"
#image_list=os.listdir(image_path)
#list_len=len(image_list)

test_img="./test.png"
test=prepare_dataset(test_img)
test=xp.array(test).astype(xp.float32).reshape(1,3,128,128)
test=chainer.as_variable(test)

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
    for batch in range(0,iterations,framesize):
        frame_box1=[]
        rnd1=np.random.randint(1,1000-framesize)
        for index in range(framesize):
            filename=image_path+"tenka_"+str(rnd1+index)+".png"
            frame=prepare_dataset(filename)
            frame_box1.append(frame)

        frames=chainer.as_variable(xp.array(frame_box1).astype(xp.float32))

        x=frames[0:framesize-1]
        t=frames[1:framesize]

        for count in range(framesize-1):
            inp=x[count].reshape(1,3,128,128)
            tar=t[count].reshape(1,3,128,128)

            y=encdec(inp)
            y_dis=discriminator(y)
            t_dis=discriminator(tar)

            y.unchain_backward()

            dis_loss=F.mean(F.softplus(y_dis))+F.mean(F.softplus(-t_dis))

            discriminator.cleargrads()
            dis_loss.backward()
            dis_opt.update()
            dis_loss.unchain_backward()

            y=encdec(inp)
            y_dis=discriminator(y)

            gen_loss=F.mean(F.softplus(-y_dis))
            gen_loss=weight*F.mean_absolute_error(y,tar)

            sum_dis_loss+=dis_loss.data.get()
            sum_gen_loss+=gen_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("encdec.model",encdec)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            for i_ in range(framesize-1):
                with chainer.using_config("train",False):
                    test=encdec(test)
                test.unchain_backward()
                y=test.data.get()
                tmp=np.clip(y[0]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(4,4,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch:{}".format(epoch))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations/framesize))
    print("EncDec loss:{}".format(sum_gen_loss/iterations/framesize))