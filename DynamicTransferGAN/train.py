import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import argparse
import os
import pylab
import cv2 as cv
from model import Discriminator,Predictor

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

def prepare_dataset(filename,size=128):
    image=cv.imread(filename)
    if image is not None:
        image = cv.resize(image,(size,size),interpolation=cv.INTER_CUBIC)
        hr_image = image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        return hr_image

def make_diff(image_array):
    source = image_array[:,0:3,:,:].reshape(1,3,size,size)
    sources = F.tile(source,(1,16,1,1))
    diff = image_array - sources

    return diff

parser=argparse.ArgumentParser(description="DynamicTransfer")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--iterations",default=5000,type=int,help="the number of iterations")
parser.add_argument("--interval",default=5,type=int,help="the interval of snapshot")
parser.add_argument("--framesize",default=16,type=int,help="frame size")
parser.add_argument("--weight",default=10.0,type=float,help="the weight of content loss")
parser.add_argument("--size",default=128,type=int,help="image width")

args=parser.parse_args()
epochs=args.epochs
iterations=args.iterations
interval=args.interval
framesize=args.framesize
weight=args.weight
size=args.size

outdir="./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

image_path="/image/"
image_list=os.listdir(image_path)
list_len=len(image_list)

input_box=[]
frame_box=[]
diff_box=[]
rnd=np.random.randint(list_len)
dir_path=image_path+image_list[rnd]
for index in range(framesize):
    inp=dir_path+"/"+str(0)+".png"
    inp=prepare_dataset(inp)
    input_box.append(inp)
    img=dir_path+"/"+str(index)+".png"
    img=prepare_dataset(img)
    diff=img-inp
    frame_box.append(img)
    diff_box.append(diff)

test_diff=chainer.as_variable(xp.concatenate(xp.array(diff_box),axis=0).astype(xp.float32)).reshape(1,48,128,128)

test_content = prepare_dataset("./test.png")
test_content = chainer.as_variable(xp.array(test_content).astype(xp.float32)).reshape(1,3,size,size)
test_content = F.tile(test_content,(1,16,1,1))

test = F.concat([test_content,test_diff],axis=1)

predictor=Predictor()
predictor.to_gpu()
pre_opt=set_optimizer(predictor)

discriminator_content=Discriminator()
discriminator_content.to_gpu()
dis_c_opt=set_optimizer(discriminator_content)

discriminator_sequence=Discriminator()
discriminator_sequence.to_gpu()
dis_s_opt=set_optimizer(discriminator_sequence)

for epoch in range(epochs):
    sum_pre_loss=0
    sum_dis_loss=0
    for batch in range(0,iterations,framesize):
        input_box=[]
        frame_box=[]
        diff_box=[]
        rnd=np.random.randint(list_len)
        dir_path=image_path+image_list[rnd]
        for index in range(framesize):
            inp=dir_path+"/"+str(0)+".png"
            inp=prepare_dataset(inp)
            input_box.append(inp)
            img=dir_path+"/"+str(index)+".png"
            img=prepare_dataset(img)
            diff=img-inp
            frame_box.append(img)
            diff_box.append(diff)

        x=chainer.as_variable(xp.concatenate(xp.array(input_box),axis=0).astype(xp.float32)).reshape(1,48,128,128)
        t=chainer.as_variable(xp.concatenate(xp.array(frame_box),axis=0).astype(xp.float32)).reshape(1,48,128,128)
        c=chainer.as_variable(xp.concatenate(xp.array(diff_box),axis=0).astype(xp.float32)).reshape(1,48,128,128)

        z=F.concat([x,c],axis=1)
        y=predictor(z)
        y_dis=discriminator_content(y)
        t_dis=discriminator_content(t)
        dis_loss=F.mean(F.softplus(-t_dis)) + F.mean(F.softplus(y_dis))

        c_g = make_diff(y)
        c_dis = discriminator_sequence(c)
        c_g_dis = discriminator_sequence(c_g)
        dis_loss+=F.mean(F.softplus(-c_dis)) + F.mean(F.softplus(c_g_dis))

        y.unchain_backward()

        discriminator_content.cleargrads()
        discriminator_sequence.cleargrads()
        dis_loss.backward()
        dis_c_opt.update()
        dis_s_opt.update()
        dis_loss.unchain_backward()

        y=predictor(z)
        y_dis=discriminator_content(y)
        t_dis=discriminator_content(t)
        gen_loss=F.mean(F.softplus(-y_dis))

        c_g = make_diff(y)
        c_dis = discriminator_sequence(c)
        c_g_dis = discriminator_sequence(c_g)
        gen_loss+=F.mean(F.softplus(-c_g_dis))

        content_loss=F.mean_absolute_error(y,t)
        content_loss+=F.mean_absolute_error(c_g,c)
        gen_loss+=weight * content_loss

        predictor.cleargrads()
        gen_loss.backward()
        pre_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += dis_loss.data.get()
        sum_pre_loss += gen_loss.data.get()

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz("predictor.model",predictor)

            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            with chainer.using_config("train",False):
                y = predictor(test)
            y = y[0].data.get()
            for i_ in range(framesize):
                tmp = np.clip(y[i_*3:(i_+1)*3]*127.5+127.5 ,0 ,255).reshape(3,128,128).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(4,4,i_+1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig("%s/visualize_%d"%(outdir,epoch))

    print("epoch : {}".format(epoch))
    print("Predictor loss : {}".format(sum_pre_loss/iterations))
    print("Discriminator loss : {}".format(sum_dis_loss/iterations))