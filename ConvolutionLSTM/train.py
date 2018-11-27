import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import argparse
import os
import pylab
import cv2 as cv
import math
from model import Predictor,Condition,Discriminator_image,Discriminator_stream
from model import FeatureEmbedding,FeatureExtractor,EncDec,VGG

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
    source = image_array[0].reshape(1,3,size,size)
    sources = F.tile(source,(framesize,1,1,1))

    return sources

parser=argparse.ArgumentParser(description="DynamicTransfer")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--iterations",default=5000,type=int,help="the number of iterations")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--framesize",default=16,type=int,help="frame size")
parser.add_argument("--weight",default=1.0,type=float,help="the weight of content loss")
parser.add_argument("--size",default=128,type=int,help="image width")

args=parser.parse_args()
epochs=args.epochs
iterations=args.iterations
interval=args.interval
framesize=args.framesize
weight=args.weight
size=args.size
wid=int(math.sqrt(framesize))

outdir="./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

image_path="/usr/MachineLearning/Dataset/cinderella/"
image_list=os.listdir(image_path)
list_len=len(image_list)

input_box=[]
frame_box=[]
rnd=np.random.randint(list_len)
dir_path=image_path+image_list[rnd]
for index in range(4,12):
    inp=dir_path+"/"+str(0)+".png"
    inp=prepare_dataset(inp)
    input_box.append(inp)
    img=dir_path+"/"+str(index)+".png"
    img=prepare_dataset(img)
    frame_box.append(img)

xtest = chainer.as_variable(xp.array(input_box).astype(xp.float32))
ctest = chainer.as_variable(xp.array(frame_box).astype(xp.float32))

test = prepare_dataset("./test.png")
test = chainer.as_variable(xp.array(test).astype(xp.float32)).reshape(1,3,size,size)
test = F.tile(test,(framesize,1,1,1))

predictor=Predictor()
predictor.to_gpu()
pre_opt=set_optimizer(predictor)

discriminator_content=Discriminator_image()
discriminator_content.to_gpu()
dis_c_opt=set_optimizer(discriminator_content)

discriminator_sequence=Discriminator_stream()
discriminator_sequence.to_gpu()
dis_s_opt=set_optimizer(discriminator_sequence)

feature_extractor=VGG()
feature_extractor.to_gpu()
fextract_opt = set_optimizer(feature_extractor)
feature_extractor.base.disable_update()

feature_embed=FeatureEmbedding()
feature_embed.to_gpu()
fembed_opt = set_optimizer(feature_embed)

for epoch in range(epochs):
    sum_pre_loss=0
    sum_dis_loss=0
    for batch in range(0,iterations,framesize):
        input_box=[]
        frame_box=[]
        rnd=np.random.randint(list_len)
        dir_path=image_path+image_list[rnd]
        for index in range(4,12):
            inp=dir_path+"/"+str(0)+".png"
            inp=prepare_dataset(inp)
            input_box.append(inp)
            img=dir_path+"/"+str(index)+".png"
            img=prepare_dataset(img)
            frame_box.append(img)

        x = chainer.as_variable(xp.array(input_box).astype(xp.float32))
        t = chainer.as_variable(xp.array(frame_box).astype(xp.float32))
        embed = feature_extractor(t) - feature_extractor(x)
        c = feature_embed(embed)
        c = F.tile(c.reshape(framesize,framesize,1,1),(1,1,128,128))

        z=F.concat([x,c],axis=1)
        y=predictor(z)
        y_dis=discriminator_content(y)
        t_dis=discriminator_content(t)
        dis_loss=F.mean(F.softplus(-t_dis)) + F.mean(F.softplus(y_dis))

        y.unchain_backward()

        c_g = feature_extractor(y) - feature_extractor(make_diff(y))
        c_g = c_g.reshape(framesize,2,128,128).transpose(1,0,2,3)
        embed = embed.reshape(framesize,2,128,128).transpose(1,0,2,3)
        c_dis = discriminator_sequence(embed)
        c_g_dis = discriminator_sequence(c_g)
        dis_loss+=F.mean(F.softplus(-c_dis)) + F.mean(F.softplus(c_g_dis))

        c_g.unchain_backward()

        discriminator_content.cleargrads()
        discriminator_sequence.cleargrads()
        dis_loss.backward()
        dis_c_opt.update()
        dis_s_opt.update()
        dis_loss.unchain_backward()

        y=predictor(z)
        y_dis=discriminator_content(y)
        gen_loss=F.mean(F.softplus(-y_dis))

        c_g = feature_extractor(y) - feature_extractor(make_diff(y))
        c_g = c_g.reshape(framesize,2,128,128).transpose(1,0,2,3)
        c_g_dis = discriminator_sequence(c_g)
        gen_loss+=F.mean(F.softplus(-c_g_dis))

        content_loss=F.mean_absolute_error(y,t)
        content_loss+=F.mean_absolute_error(c_g,embed)
        gen_loss+=weight * content_loss

        predictor.cleargrads()
        feature_extractor.cleargrads()
        feature_embed.cleargrads()
        gen_loss.backward()
        pre_opt.update()
        fextract_opt.update()
        fembed_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += dis_loss.data.get()
        sum_pre_loss += gen_loss.data.get()

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz("predictor.model",predictor)

            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            with chainer.using_config("train",False):
                c = feature_extractor(ctest) - feature_extractor(xtest)
                c = feature_embed(c)
                c = F.tile(c.reshape(framesize,framesize,1,1),(1,1,128,128))
                y = predictor(F.concat([test,c]))
            c.unchain_backward()
            y.unchain_backward()
            y = y.data.get()
            for i_ in range(framesize):
                tmp = np.clip(y[i_]*127.5+127.5 ,0 ,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(wid,int(framesize/wid),i_+1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig("%s/visualize_%d"%(outdir,epoch))

    print("epoch : {}".format(epoch))
    print("Predictor loss : {}".format(sum_pre_loss/iterations))
    print("Discriminator loss : {}".format(sum_dis_loss/iterations))