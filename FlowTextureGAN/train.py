import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
import cv2 as cv
from model import Encoder,Refine,Discriminator
from prepare import prepare_dataset,prepare_test,optical_flow,prepare_image
import requests

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description="TFGAN")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--interval",default=5,type=int,help="the interval of snapshot")
parser.add_argument("--framesize",default=16,type=int,help="frame size")
parser.add_argument("--weight",default=10.0,type=float,help="the weight of content losss")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iterations")

args=parser.parse_args()
epochs=args.epochs
interval=args.interval
framesize=args.framesize
weight=args.weight
iterations=args.iterations

outdir="./output_gan/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

image_path="/image/"
image_list=os.listdir(image_path)
image_len=len(image_list)

rnd=np.random.randint(image_len)
dir_path=image_path+image_list[rnd]
print(dir_path)
left_of_box=[]
left_input_box=[]
for index in range(framesize):
    filename1=dir_path+"/lefteye_"+str(0)+".png"
    filename2=dir_path+"/lefteye_"+str(index)+".png"
    ref=optical_flow(filename1,filename2)
    left_of_box.append(ref)

right_of_box=[]
right_input_box=[]
for index in range(framesize):
    filename1=dir_path+"/righteye_"+str(0)+".png"
    filename2=dir_path+"/righteye_"+str(index)+".png"
    ref=optical_flow(filename1,filename2)
    right_of_box.append(ref)

lotest=chainer.as_variable(xp.array(left_of_box).astype(xp.float32))
rotest=chainer.as_variable(xp.array(right_of_box).astype(xp.float32))

test_path="./test.png"
test,lefteye,leftlist,righteye,rightlist=prepare_test(test_path)
left=chainer.as_variable(xp.array(lefteye).astype(xp.float32)).reshape(1,3,32,32)
right=chainer.as_variable(xp.array(righteye).astype(xp.float32)).reshape(1,3,32,32)
left=F.tile(left,(framesize,1,1,1))
right=F.tile(right,(framesize,1,1,1))

encoder=Encoder()
encoder.to_gpu()
enc_opt=set_optimizer(encoder)

refine=Refine()
refine.to_gpu()
ref_opt=set_optimizer(refine)

discriminator=Discriminator()
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,iterations,framesize):
        input_box=[]
        target_box=[]
        opt_box=[]
        rnd=np.random.randint(image_len)
        dir_path=image_path+image_list[rnd]
        ta=np.random.choice(["lefteye","righteye"])
        for index in range(framesize):
            filename1=dir_path+"/"+ta+"_"+str(0)+".png"
            inp=prepare_dataset(filename1)
            input_box.append(inp)
            filename2=dir_path+"/"+ta+"_"+str(index)+".png"
            img=prepare_dataset(filename2)
            target_box.append(img)
            ref=optical_flow(filename1,filename2)
            opt_box.append(ref)

        x=chainer.as_variable(xp.array(input_box).astype(xp.float32))
        t=chainer.as_variable(xp.array(target_box).astype(xp.float32))
        opt=chainer.as_variable(xp.array(opt_box).astype(xp.float32))

        #y=encoder(F.concat([x,opt]))

        #_, channels, height, width=y.shape
        #y=y.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)
        #opt3=opt.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)
        #y=refine(y)

        #t=t.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)

        #y_dis=discriminator(y)
        #t_dis=discriminator(t)
        #dis_loss=F.mean(F.softplus(-t_dis)) + F.mean(F.softplus(y_dis))

        #discriminator.cleargrads()
        #dis_loss.backward()
        #dis_opt.update()
        #dis_loss.unchain_backward()

        y=encoder(F.concat([x,opt]))

        _, channels, height, width=y.shape
        y=y.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)
        opt3=opt.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)
        y=refine(y)

        t=t.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)
        gen_loss=F.mean_absolute_error(y,t)
        #y_dis = discriminator(y)
        #gen_loss+=F.mean(F.softplus(-y_dis))

        encoder.cleargrads()
        #decoder.cleargrads()
        #refine.cleargrads()

        gen_loss.backward()

        enc_opt.update()
        #dec_opt.update()
        #ref_opt.update()

        gen_loss.unchain_backward()

        #for p in discriminator.params():
        #    p.data = xp.clip(p.data,-0.01,0.01)

        sum_gen_loss+=gen_loss.data.get()
        #sum_dis_loss+=dis_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("encoder_gan.model",encoder)
            #serializers.save_npz("decoder.model",decoder)
            serializers.save_npz("refine_gan.model",refine)

            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()

            l=encoder(F.concat([left,lotest]))
            #l=l.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)
            #lo=lotest.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)
            #l=refine(l)
            l.unchain_backward()
            #l=l[0].transpose(1,0,2,3)
            l=l.data.get()

            r=encoder(F.concat([right,rotest]))
            #r=r.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)
            #ro=rotest.reshape(1,framesize,channels,height,width).transpose(0,2,1,3,4)
            #r=refine(r)
            r.unchain_backward()
            #r=r[0].transpose(1,0,2,3)
            r=r.data.get()

            for i_ in range(framesize):
                tmp_left=np.clip(l[i_]*127.5+127.5,0,255).transpose(1,2,0).astype(xp.uint8)
                tmp_right=np.clip(r[i_]*127.5+127.5,0,255).transpose(1,2,0).astype(xp.uint8)
                tmp_left=cv.resize(tmp_left,(30,30),interpolation=cv.INTER_CUBIC)
                tmp_right=cv.resize(tmp_right,(30,30),interpolation=cv.INTER_CUBIC)
                test[leftlist[0]:leftlist[1], leftlist[2]:leftlist[3]] = tmp_left
                test[rightlist[0]:rightlist[1], rightlist[2]:rightlist[3]] = tmp_right
                pylab.subplot(4,4,i_+1)
                pylab.imshow(test)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch:{}".format(epoch))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))
    print("Generator loss:{}".format(sum_gen_loss/iterations))