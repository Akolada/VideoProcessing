import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
import cv2 as cv
from model import ImageEncoder, UNet, Discriminator_temporal, Discriminator_image, KeyPointDetector, Generator
from prepare import prepare_dataset,prepare_test,optical_flow,prepare_image

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

def preapre_smoothing(optical_flow, frames):
    opts = None
    for index in range(1, framesize):
        opt_prev = optical_flow[index - 1]
        opt_after = optical_flow[index]
        opt_concat = F.concat([opt_prev, opt_after], axis=0)

        frame_prev = frames[index - 1]
        frame_after = frames[index]
        frame_concat = F.concat([frame_prev, frame_after], axis=0)

        concat = F.concat([opt_concat, frame_concat], axis=0).reshape(1,12,128,128)
        if opts is None:
            opts = concat
        else:
            opts = F.concat([opts, concat], axis=0)

    return opts

parser=argparse.ArgumentParser(description="AddMotion")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--interval",default=5,type=int,help="the interval of snapshot")
parser.add_argument("--framesize",default=12,type=int,help="frame size")
parser.add_argument("--weight",default=10.0,type=float,help="the weight of content losss")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iterations")

args=parser.parse_args()
epochs=args.epochs
interval=args.interval
framesize=args.framesize
weight=args.weight
iterations=args.iterations

outdir="./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

image_path="./syani"
image_list=os.listdir(image_path)
image_len=len(image_list)

rnd=np.random.randint(image_len)
test_input_box =[]
test_of_box=[]
test_box=[]
rnd = np.random.randint(image_len)
dir_path = image_path + image_list[rnd]
start_frame = np.random.randint(1, 200 - framesize)
for index in range(framesize):
    filename1=dir_path+"/face_"+str(start_frame)+".png"
    filename2=dir_path+"/face_"+str(start_frame + index)+".png"
    inp = prepare_image(filename1)
    test_input_box.append(inp)
    img = prepare_image(filename2)
    test_box.append(img)
    ref=optical_flow(filename1,filename2)
    test_of_box.append(ref)

pylab.rcParams['figure.figsize'] = (16.0,16.0)
pylab.clf()

test_opt=chainer.as_variable(xp.array(test_of_box).astype(xp.float32))
test_input = chainer.as_variable(xp.array(test_input_box).astype(xp.float32))
test_frame = chainer.as_variable(xp.array(test_box).astype(xp.float32))

for i, t in enumerate(test_box):
    tmp=np.clip(t*127.5+127.5,0,255).transpose(1,2,0).astype(xp.uint8)
    pylab.subplot(4,4,i+1)
    pylab.imshow(tmp)
    pylab.axis('off')
    pylab.savefig('%s/true.png'%(outdir))

test_path="./test.png"
test=prepare_image(test_path)
test=chainer.as_variable(xp.array(test).astype(xp.float32))
test=F.tile(test,(framesize,1,1,1))

image_encoder=ImageEncoder()
image_encoder.to_gpu()
enc_opt=set_optimizer(image_encoder)

key_point_detector = KeyPointDetector()
key_point_detector.to_gpu()
key_opt = set_optimizer(key_point_detector)

making_optical_flow = UNet(in_ch=4)
making_optical_flow.to_gpu()
ref_opt=set_optimizer(making_optical_flow)

generator = Generator(in_ch=3)
generator.to_gpu()
gen_opt = set_optimizer(generator)

discriminator_temporal = Discriminator_temporal()
discriminator_temporal.to_gpu()
dis_temp_opt=set_optimizer(discriminator_temporal)

discriminator_image = Discriminator_image()
discriminator_image.to_gpu()
dis_img_opt = set_optimizer(discriminator_image)

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,iterations,framesize):
        input_box=[]
        target_box=[]
        opt_box=[]
        rnd = np.random.randint(image_len)
        dir_path = image_path + image_list[rnd]
        start_frame = np.random.randint(1, 200 - framesize)
        for index in range(framesize):
            filename1=dir_path+"/face_"+str(start_frame)+".png"
            inp=prepare_image(filename1)
            input_box.append(inp)
            filename2=dir_path+"/face_"+str(start_frame + index)+".png"
            img=prepare_image(filename2)
            target_box.append(img)
            ref=optical_flow(filename1,filename2)
            opt_box.append(ref)

        x=chainer.as_variable(xp.array(input_box).astype(xp.float32))
        t=chainer.as_variable(xp.array(target_box).astype(xp.float32))
        opt=chainer.as_variable(xp.array(opt_box).astype(xp.float32))

        key_x = key_point_detector(x)
        key_t = key_point_detector(t)
        key_diff = key_t - key_x

        opt_enc=image_encoder(opt)
        opt_fake = making_optical_flow(F.concat([x, key_diff]), opt_enc)
        opt_fake_enc = image_encoder(opt_fake)
        y = generator(x, key_diff, opt_fake_enc)

        y.unchain_backward()

        dis_opt_real = discriminator_image(F.concat([x, opt]))
        dis_opt_fake = discriminator_image(F.concat([x, opt_fake]))

        temp_fake = preapre_smoothing(opt_fake, y)
        temp_real = preapre_smoothing(opt_fake, t)

        dis_temp_real = discriminator_temporal(temp_real)
        dis_temp_fake = discriminator_temporal(temp_fake)

        dis_optical = F.mean(F.softplus(-dis_opt_real)) + F.mean(F.softplus(dis_opt_fake))
        dis_temp = F.mean(F.softplus(-dis_temp_real)) + F.mean(F.softplus(dis_temp_fake))

        dis_loss = dis_temp + dis_optical

        discriminator_image.cleargrads()
        discriminator_temporal.cleargrads()
        dis_loss.backward()
        dis_img_opt.update()
        dis_temp_opt.update()
        dis_loss.unchain_backward()

        key_x = key_point_detector(x)
        key_t = key_point_detector(t)
        key_diff = key_t - key_x

        opt_enc=image_encoder(opt)
        opt_fake = making_optical_flow(F.concat([x, key_diff]), opt_enc)
        opt_fake_enc = image_encoder(opt_fake)
        y = generator(x, key_diff, opt_fake_enc)

        dis_opt_fake = discriminator_image(F.concat([x, opt_fake]))
        temp_fake = preapre_smoothing(opt_fake, y)
        dis_temp_fake = discriminator_temporal(temp_fake)

        gen_loss = F.mean(F.softplus(-dis_temp_fake)) + F.mean(F.softplus(-dis_opt_fake))
        gen_loss += F.mean_absolute_error(y,t) + F.mean_absolute_error(opt_fake, opt)

        key_point_detector.cleargrads()
        image_encoder.cleargrads()
        making_optical_flow.cleargrads()
        generator.cleargrads()
        gen_loss.backward()
        enc_opt.update()
        ref_opt.update()
        gen_opt.update()
        key_opt.update()

        gen_loss.unchain_backward()
        
        sum_gen_loss+=gen_loss.data.get()
        sum_dis_loss += dis_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("image_encoder.model",image_encoder)
            serializers.save_npz("making_opticalflow.model",making_optical_flow)
            serializers.save_npz("generator.model",generator)

            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()

            with chainer.using_config('train', False):
                key_x = key_point_detector(test_input)
                key_t = key_point_detector(test_frame)
                key_diff = key_t - key_x

                opt_enc=image_encoder(test_opt)
                opt_fake = making_optical_flow(F.concat([test, key_diff]), opt_enc)
                opt_fake_enc = image_encoder(opt_fake)
                y = generator(test, key_diff, opt_fake_enc)

                y.unchain_backward()

            y = y.data.get()
            opt_fake = opt_fake.data.get()
            key_diff = key_diff.data.get()
            print(key_diff.shape)

            for i_ in range(framesize):
                tmp=np.clip(key_diff[i_][0]*255.0,0,255).astype(np.uint8)
                pylab.subplot(4,4,i_+1)
                pylab.imshow(tmp,cmap='gray')
                pylab.gray()
                pylab.axis('off')
                pylab.savefig('%s/keypoint_diff_%d.png'%(outdir, epoch))

            for i_ in range(framesize):
                tmp=np.clip(opt_fake[i_]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(4,4,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/opticalflow_%d.png'%(outdir, epoch))

            for i_ in range(framesize):
                tmp=np.clip(y[i_]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(4,4,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch:{}".format(epoch))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))
    print("Generator loss:{}".format(sum_gen_loss/iterations))