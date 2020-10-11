# coding=utf-8

import os, time, argparse, network, util
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import dlib
import numpy as np
#from torch.nn.parallel import DistributedDataParallel
from torchvision.models import vgg16

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# torch.distributed.init_process_group(backend="nccl")
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
device = torch.device("cuda:0")#, local_rank)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='img',  help='')
parser.add_argument('--train_subfolder', required=False, default='train',  help='')
parser.add_argument('--test_subfolder', required=False, default='test',  help='')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--nb', type=int, default=9, help='the number of resnet block layer for generator')
parser.add_argument('--train_epoch', type=int, default=200, help='train epochs num')
parser.add_argument('--decay_epoch', type=int, default=30, help='learning rate decay start epoch num')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
parser.add_argument('--lambda_eye', type=float, default=8, help='lambdaA for cycle loss')
parser.add_argument('--lambda_lip', type=float, default=5, help='lambdaB for cycle loss')
parser.add_argument('--lambda_face', type=float, default=3, help='lambdaB for cycle loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--save_root', required=False, default='results', help='results save path')
# parser.add_argument('--local_rank')

opt = parser.parse_args()
print("type=",type(opt))
print('------------ Options -------------')
for k, v in sorted(vars(opt).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# results save path
root = '/data/home/yang-bj/collab_beauty/Plan1_Multi-4-MobileNet_v1/'+ opt.dataset + '_' + opt.save_root + '/' #apple2orange2_results/
model = opt.dataset + '_'
if not os.path.isdir(root):
    os.mkdir(root)

# data_loader
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
path_data_root = '/data/home/yang-bj/collab_beauty/MobileNet_beautyGAN/'
train_loader_A = util.data_load(path_data_root+ opt.dataset, opt.train_subfolder + 'A', transform, opt.batch_size, shuffle=True)#.cpu()
train_loader_B = util.data_load(path_data_root+ opt.dataset, opt.train_subfolder + 'B', transform, opt.batch_size, shuffle=True)#.cpu()
test_loader_A = util.data_load(path_data_root+ opt.dataset, opt.test_subfolder + 'A', transform, opt.batch_size, shuffle=False)#.cpu()
test_loader_B = util.data_load(path_data_root+ opt.dataset, opt.test_subfolder + 'B', transform, opt.batch_size, shuffle=False)#.cpu()

# network
B_encoder = network.Big_encoder()
B_decoder = network.Big_decoder()
Gs = network.generator_encoder_Small()
D_A = network.discriminator()
D_B = network.discriminator()
G_state_dict = torch.load('/data/home/yang-bj/collab_beauty/Plan1_Multi-4-MobileNet_v1/model/img_generator_param_decompose.pkl')
D_A_dict = torch.load('/data/home/yang-bj/collab_beauty/Plan1_Multi-4-MobileNet_v1/model/img_discriminatorA_param_decompose.pkl')
D_B_dict = torch.load('/data/home/yang-bj/collab_beauty/Plan1_Multi-4-MobileNet_v1/model/img_discriminatorB_param_decompose.pkl')
D_A_state_dict = D_A.state_dict()
D_B_state_dict = D_B.state_dict()
for k,v in D_A_state_dict.items():
    D_A_state_dict[k].copy_(D_A_dict['module.'+k])
for k,v in D_B_state_dict.items():
    D_B_state_dict[k].copy_(D_B_dict['module.'+k])

model_dict_de =  B_decoder.state_dict()
model_dict_en =  B_encoder.state_dict()
state_dict_de = {k[7:]:v for k,v in G_state_dict.items() if k[7:] in model_dict_de.keys()}
model_dict_de.update(state_dict_de)
B_decoder.load_state_dict(model_dict_de)

keys_B_en = list(model_dict_en)
keys_G = list(G_state_dict)[0:len(keys_B_en)]
i = 0
while i < len(keys_B_en):
    model_dict_en[keys_B_en[i]].copy_(G_state_dict[keys_G[i]])
    i+=1
B_encoder.load_state_dict(model_dict_en)

Gs.to(device)
D_A.to(device)
D_B.to(device)
B_decoder.to(device)
B_encoder.to(device)
# D_A = DistributedDataParallel(D_A, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
# D_B = DistributedDataParallel(D_B,find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
# Gs = DistributedDataParallel(Gs,find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
# B_decoder = DistributedDataParallel(B_decoder,find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
# B_encoder = DistributedDataParallel(B_encoder,find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)

B_decoder.eval()
B_encoder.eval()
Gs.train()
D_A.eval()
D_B.eval()

def decoder_forward(model, x):
    x,y = model(x)
    return x,y

# loss
BCE_loss = nn.BCELoss().to(device)
MSE_loss = nn.MSELoss().to(device)
L1_loss = nn.L1Loss().to(device)

vgg = vgg16(pretrained=False)
vgg.load_state_dict(torch.load(path_data_root + 'model/vgg16-397923af.pth'))
vgg = vgg.features[:18]
vgg.to(device)
vgg.eval()
# vgg = DistributedDataParallel(vgg,find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)

# Face mask model
predictor = dlib.shape_predictor(
        path_data_root + "model/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Adam optimizer
Gs_optimizer = optim.Adam(Gs.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))

# image store
fakeA_store = util.ImagePool(50)
fakeB_store = util.ImagePool(50)

train_hist = {}
train_hist['G_A_losses'] = []
train_hist['G_B_losses'] = []
train_hist['A_cycle_losses'] = []
train_hist['B_cycle_losses'] = []
train_hist['histogram_losses'] = []
train_hist['perceptual_losses'] = []
train_hist['idt_losses'] = []
train_hist['feat_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

minloss = 100
print('training start!')
start_time = time.time()
torch.backends.cudnn.benchmark = True
for epoch in range(opt.train_epoch):
    G_A_losses = []
    G_B_losses = []
    A_cycle_losses = []
    B_cycle_losses = []
    histogram_losses = []
    perceptual_losses = []
    idt_losses = []
    feat_losses = []
    G_losses = []
    epoch_start_time = time.time()
    iter = 0

    if (epoch+1) > opt.decay_epoch: #start dacay lr
        Gs_optimizer.param_groups[0]['lr'] -= opt.lrG / (opt.train_epoch - opt.decay_epoch)

    for (img_A, _), (img_B, _) in zip(train_loader_A, train_loader_B):
        torch.cuda.empty_cache()

        realA = util.to_var(img_A, requires_grad=False)
        realB = util.to_var(img_B, requires_grad=False)


        resA = (realA.cpu().squeeze(0).permute(1, 2, 0).data.numpy()+1) * 127.5
        resB = (realB.cpu().squeeze(0).permute(1, 2, 0).data.numpy()+1) * 127.5

        input_A_mask = np.zeros((opt.batch_size, 4, 256, 256))
        res = util.get_mask(resA.astype(np.uint8), detector, predictor)
        if res is not None:
            input_A_mask[0][0] = np.equal(res[0], 255)
            input_A_mask[0][1] = np.equal(res[1], 255)
            input_A_mask[0][2] = np.equal(res[2], 255)
            input_A_mask[0][3] = np.equal(res[3], 255)
        else: continue
        input_A_mask = Variable(torch.as_tensor(input_A_mask).to(device),requires_grad=False)

        input_B_mask = np.zeros((opt.batch_size, 4, 256, 256))
        res = util.get_mask(resB.astype(np.uint8), detector, predictor)
        if res is not None:
            input_B_mask[0][0] = np.equal(res[0], 255)
            input_B_mask[0][1] = np.equal(res[1], 255)
            input_B_mask[0][2] = np.equal(res[2], 255)
            input_B_mask[0][3] = np.equal(res[3], 255)
        else:continue
        input_B_mask = Variable(torch.as_tensor(input_B_mask).to(device), requires_grad=False)
        # print("------------mask loaded-----------------")

        ##########################################################
        #################      train generator        ############
        ##########################################################
        torch.cuda.empty_cache()
        # middle_res = B_encoder(realA, realB)
        # fakeB_old, fakeA_old = B_decoder(torch.cat((middle_res[1],middle_res[3]),dim=1))
        # del middle_res

        out_Gs = Gs(realA, realB)
        out_En = B_encoder(realA, realB)

        feat_loss = 0
        for i in range(len(out_En)):
            feat_loss += nn.MSELoss()(out_Gs[i], out_En[i])
        fakeB, fakeA = decoder_forward(B_decoder, out_Gs[-1])
        #fakeB_big, fakeA_big = decoder_forward(B_decoder, torch.cat((out_En[1], out_En[3]),dim=1))
        del out_Gs,out_En
        torch.cuda.empty_cache()

        out_Gs = Gs(fakeA, fakeB)
        out_En = B_encoder(fakeA, fakeB)
        feat_loss_rec = 0
        for i in range(len(out_En)):
            feat_loss_rec += nn.MSELoss()(out_Gs[i], out_En[i])
        recB, recA = decoder_forward(B_decoder, out_Gs[-1])
        del out_Gs, out_En
        torch.cuda.empty_cache()



        feat_loss_sum = feat_loss + feat_loss_rec
        #del feat_loss, feat_loss_rec
        torch.cuda.empty_cache()

        Gs_optimizer.zero_grad()

        D_B_result = D_B(fakeB) 
        G_A_loss = MSE_loss(D_B_result, Variable(torch.ones(D_B_result.size()).to(device)))
        del D_B_result
        torch.cuda.empty_cache()

        A_cycle_loss = L1_loss(recA, realA) * opt.lambdaA

        D_A_result = D_A(fakeA)
        G_B_loss = MSE_loss(D_A_result, Variable(torch.ones(D_A_result.size()).to(device)))
        del D_A_result
        torch.cuda.empty_cache()

        B_cycle_loss = L1_loss(recB, realB) * opt.lambdaB

        #########################################################################################
        ############                  make up loss                         ######################
        ############                 perceptual loss                       ######################
        #########################################################################################
        #######################          r           ###############################
        torch.cuda.empty_cache()

        histogram_loss_lip = 0
        histogram_loss_eye = 0
        histogram_loss_eyebrow  = 0
        histogram_loss_face = 0
        histogram_loss = 0


        for cur in range(opt.batch_size):
            temp_source = ((fakeB[cur, 0, :, :] + 1) * 127.5).to(torch.float32).to(device)
            temp_template = ((realB[cur, 0, :, :] + 1) * 127.5).to(torch.float32).to(device)
            histogram_loss_lip += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][0],
                                                           input_B_mask[cur][0])*opt.lambda_lip
            histogram_loss_eye += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][1],
                                                           input_B_mask[cur][1])*opt.lambda_eye
            histogram_loss_eyebrow += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][2],
                                                          input_B_mask[cur][2])
            histogram_loss_face += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][3],
                                                          input_B_mask[cur][3])*opt.lambda_face
            del temp_source, temp_template
            torch.cuda.empty_cache()
            #####################            g             #######################33
            temp_source = ((fakeB[cur, 1, :, :] + 1) * 127.5).to(torch.float32).to(device)
            temp_template = ((realB[cur, 1, :, :] + 1) * 127.5).to(torch.float32).to(device)
            histogram_loss_lip += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][0],
                                                           input_B_mask[cur][0])*opt.lambda_lip
            histogram_loss_eye += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][1],
                                                           input_B_mask[cur][1])*opt.lambda_eye
            histogram_loss_eyebrow += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][2],
                                                              input_B_mask[cur][2])
            histogram_loss_face += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][3],
                                                           input_B_mask[cur][3]) *opt.lambda_face
            del temp_source, temp_template
            torch.cuda.empty_cache()
            ######################           b            ########################
            temp_source = ((fakeB[cur, 2, :, :] + 1) * 127.5).to(torch.float32).to(device)
            temp_template = ((realB[cur, 2, :, :] + 1) * 127.5).to(torch.float32).to(device)

            histogram_loss_lip += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][0],
                                                           input_B_mask[cur][0])*opt.lambda_lip
            histogram_loss_eye += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][1],
                                                           input_B_mask[cur][1])*opt.lambda_eye
            histogram_loss_eyebrow += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][2],
                                                              input_B_mask[cur][2])
            histogram_loss_face += util.histogram_loss_cal(temp_source, temp_template, input_A_mask[cur][3],
                                                           input_B_mask[cur][3])*opt.lambda_face
            del temp_source, temp_template
            torch.cuda.empty_cache()
            #######################           r                         #############################
            temp_source = ((fakeA[cur, 0, :, :] + 1) * 127.5).to(torch.float32).to(device)
            temp_template = ((realA[cur, 0, :, :] + 1) * 127.5).to(torch.float32).to(device)
            histogram_loss_lip += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][0],
                                                          input_A_mask[cur][0]) * opt.lambda_lip
            histogram_loss_eye += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][1],
                                                          input_A_mask[cur][1]) * opt.lambda_eye
            histogram_loss_eyebrow += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][2],
                                                              input_A_mask[cur][2])
            histogram_loss_face += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][3],
                                                           input_A_mask[cur][3]) * opt.lambda_face
            del temp_source, temp_template
            torch.cuda.empty_cache()
            #####################            g             #######################33
            temp_source = ((fakeA[cur, 1, :, :] + 1) * 127.5).to(torch.float32).to(device)
            temp_template = ((realA[cur, 1, :, :] + 1) * 127.5).to(torch.float32).to(device)
            histogram_loss_lip += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][0],
                                                          input_A_mask[cur][0]) * opt.lambda_lip
            histogram_loss_eye += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][1],
                                                          input_A_mask[cur][1]) * opt.lambda_eye
            histogram_loss_eyebrow += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][2],
                                                              input_A_mask[cur][2])
            histogram_loss_face += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][3],
                                                           input_A_mask[cur][3]) * opt.lambda_face
            del temp_source, temp_template
            torch.cuda.empty_cache()
            ######################           b            ########################
            temp_source = ((fakeA[cur, 2, :, :] + 1) * 127.5).to(torch.float32).to(device)
            temp_template = ((realA[cur, 2, :, :] + 1) * 127.5).to(torch.float32).to(device)

            histogram_loss_lip += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][0],
                                                          input_A_mask[cur][0]) * opt.lambda_lip
            histogram_loss_eye += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][1],
                                                          input_A_mask[cur][1]) * opt.lambda_eye
            histogram_loss_eyebrow += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][2],
                                                              input_A_mask[cur][2])
            histogram_loss_face += util.histogram_loss_cal(temp_source, temp_template, input_B_mask[cur][3],
                                                           input_A_mask[cur][3]) * opt.lambda_face
            del temp_source, temp_template
            torch.cuda.empty_cache()
        histogram_loss = (histogram_loss_lip + histogram_loss_eye + histogram_loss_eyebrow + histogram_loss_face)*0.5*0.1
        del histogram_loss_lip, histogram_loss_eye, histogram_loss_eyebrow, histogram_loss_face
        torch.cuda.empty_cache()

        del input_A_mask, input_B_mask
        torch.cuda.empty_cache()

        histogram_loss = histogram_loss.to(device)

        torch.cuda.empty_cache()

        up = nn.Upsample(size=224, mode='bilinear')
        perc_A = up((realA + 1) * 127.5).to(torch.float32)
        perc_fake_B = up((fakeB + 1) * 127.5).to(torch.float32)
        perc_B = up((realB + 1) * 127.5).to(torch.float32)
        perc_fake_A = up((fakeA + 1) * 127.5).to(torch.float32)


        perc_A = perc_A.detach()
        perc_fake_B = perc_fake_B.detach()
        perc_B = perc_B.detach()
        perc_fake_A = perc_fake_A.detach()


        # input into the pretrained VGG16 and standardize
        perc = vgg(torch.cat([perc_A, perc_B, perc_fake_B, perc_fake_A,], axis=0))
        # perc = vgg.features[:18](torch.cat([perc_A, perc_B, perc_fake_B, perc_fake_A,
        #                   cyc_perc_A, cyc_perc_B, cyc_perc_fake_B, cyc_perc_fake_A], axis=0))  # 18-->conv4_1
        del perc_A, perc_B, perc_fake_B, perc_fake_A
        torch.cuda.empty_cache()
        perc_mean = torch.mean(perc)

        perc = torch.div(perc, torch.add(perc_mean, 1e-5))
        del perc_mean
        torch.cuda.empty_cache()

        perceptual_loss = (torch.mean(torch.pow(torch.sub(perc[0], perc[2]), 2)) +
                          torch.mean(torch.pow(torch.sub(perc[1], perc[3]), 2)) ).to(device)*0.025

        out_Gs = Gs(realA, realA)
        idt_A1, idt_A2 = decoder_forward(B_decoder, out_Gs[-1])
        out_Gs = Gs(realB, realB)
        idt_B1, idt_B2 = decoder_forward(B_decoder, out_Gs[-1])
        idt_loss_A1 = L1_loss(idt_A1, realA)
        idt_loss_A2 = L1_loss(idt_A2, realA)
        idt_loss_B1 = L1_loss(idt_B1, realB)
        idt_loss_B2 = L1_loss(idt_B2, realB)
        del idt_A1, idt_A2, idt_B1, idt_B2
        torch.cuda.empty_cache()
        idt_loss = (idt_loss_A1 + idt_loss_A2 + idt_loss_B1 + idt_loss_B2) * opt.lambdaA * 0.5 * 0.5
        del idt_loss_A1, idt_loss_A2, idt_loss_B1, idt_loss_B2
        torch.cuda.empty_cache()



        G_loss = G_A_loss + G_B_loss + A_cycle_loss + B_cycle_loss + histogram_loss + perceptual_loss + idt_loss + feat_loss_sum
        del realA, realB, fakeA, fakeB, recA, recB
        torch.cuda.empty_cache()

        G_loss.backward(retain_graph=True)
        Gs_optimizer.step()

        train_hist['G_A_losses'].append(G_A_loss.item())
        train_hist['G_B_losses'].append(G_B_loss.item())
        train_hist['A_cycle_losses'].append(A_cycle_loss.item())
        train_hist['B_cycle_losses'].append(B_cycle_loss.item())
        train_hist['histogram_losses'].append(histogram_loss.item())
        train_hist['perceptual_losses'].append(perceptual_loss.item())
        train_hist['idt_losses'].append(idt_loss.item())
        train_hist['feat_losses'].append(feat_loss_sum.item())
        train_hist['G_losses'].append(G_loss.item())

        G_A_losses.append(G_A_loss.item())
        G_B_losses.append(G_B_loss.item())
        A_cycle_losses.append(A_cycle_loss.item())
        B_cycle_losses.append(B_cycle_loss.item())
        histogram_losses.append(histogram_loss.item())
        perceptual_losses.append(perceptual_loss.item())
        idt_losses.append(idt_loss.item())
        feat_losses.append(feat_loss_sum.item())
        G_losses.append(G_loss.item())

        iter += 1
        if iter > 2000: break
        if iter%50==0:
            print(
                'iter : %d , loss_G_A: %.3f, loss_G_B: %.3f,\n '
                'loss_A_cycle: %.3f, loss_B_cycle: %.3f, histogram_loss: %.3f, '
                'perceptual_loss: %.3f, idt_loss: %.3f, G_loss: %.3f' % (
                    (iter + 1),  torch.mean(torch.FloatTensor(G_A_losses)),
                    torch.mean(torch.FloatTensor(G_B_losses)), torch.mean(torch.FloatTensor(A_cycle_losses)),
                    torch.mean(torch.FloatTensor(B_cycle_losses)), torch.mean(torch.FloatTensor(histogram_losses)),
                    torch.mean(torch.FloatTensor(perceptual_losses)), torch.mean(torch.FloatTensor(idt_losses)),
                    torch.mean(torch.FloatTensor(G_losses))))
        del G_A_loss, G_B_loss, A_cycle_loss, B_cycle_loss, histogram_loss, perceptual_loss, idt_loss, feat_loss_sum
        torch.cuda.empty_cache()
        if iter%10==0:
            print("iter=",iter)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    print(
        '[%d/%d] - ptime: %.2f,  loss_G_A: %.3f, loss_G_B: %.3f,\n '
        'loss_A_cycle: %.3f, loss_B_cycle: %.3f, histogram_loss: %.3f, '
        'perceptual_loss: %.3f, idt_loss: %.3f,  G_loss: %.3f' % (
            (epoch + 1), opt.train_epoch, per_epoch_ptime,  torch.mean(torch.FloatTensor(G_A_losses)),
            torch.mean(torch.FloatTensor(G_B_losses)), torch.mean(torch.FloatTensor(A_cycle_losses)),
            torch.mean(torch.FloatTensor(B_cycle_losses)), torch.mean(torch.FloatTensor(histogram_losses)),
            torch.mean(torch.FloatTensor(perceptual_losses)), torch.mean(torch.FloatTensor(idt_losses)),
            torch.mean(torch.FloatTensor(G_losses))))
    
    if not os.path.isdir(root + 'epoch' + str(epoch)):
        os.mkdir(root + 'epoch' + str(epoch))
    if (epoch + 1) % 1 == 0 or torch.mean(torch.FloatTensor(G_losses))< minloss:
        if (epoch + 1) % 10 == 0 or torch.mean(torch.FloatTensor(G_losses))< minloss:
            minloss = torch.mean(torch.FloatTensor(G_losses))
            torch.save(Gs.state_dict(), root + 'epoch' + str(epoch) + '/' + model + 'generatorSmall_param.pkl')
            if epoch == 0:
                torch.save(B_decoder.state_dict(), root + 'epoch' + str(epoch) + '/' + model + 'B_decoder.pkl')
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!small!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",minloss)
        # test A to B
        print("save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!test")
        iter = 0
        for (img_A, _), (img_B, _) in zip(test_loader_A, test_loader_B):
            iter += 1
            realA = util.to_var(img_A, requires_grad=False)
            realB = util.to_var(img_B, requires_grad=False)
            middle_res = B_encoder(realA, realB)
            genB, genA = B_decoder(torch.cat((middle_res[1], middle_res[3]), dim=1))
            # genB, genA = B_decoder(torch.cat((B_encoder(realA, realB)),dim=1))#G(realA, realB)
            out_Gs = Gs(realA, realB)
            #out_En = B_encoder(realA, realB)
            fakeB, fakeA = decoder_forward(B_decoder, out_Gs[-1])
            # fakeB_big, fakeA_big = decoder_forward(B_decoder, torch.cat((out_En[1], out_En[3]),dim=1))

            out_Gs = Gs(fakeA, fakeB)
            # out_En = B_encoder(fakeA, fakeB)
            recB, recA = decoder_forward(B_decoder, out_Gs[-1])

            if not os.path.isdir(root + 'epoch' + str(epoch) + '/' + 'test_results'):
                os.mkdir(root + 'epoch' + str(epoch) + '/' + 'test_results')

            if not os.path.isdir(root + 'epoch' + str(epoch) + '/' + 'test_results/AtoB'):
                os.mkdir(root + 'epoch' + str(epoch) + '/' + 'test_results/AtoB')
            path = root + 'epoch' + str(epoch) + '/' + 'test_results/AtoB/' + str(iter) + '_input.png'
            plt.imsave(path, (realA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'test_results/AtoB/' + str(iter) + 'genB.png'
            plt.imsave(path, (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'test_results/AtoB/' + str(iter) + 'fakeB.png'
            plt.imsave(path, (fakeB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'test_results/AtoB/' + str(iter) + '_recon.png'
            plt.imsave(path, (recA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

            if not os.path.isdir(root + 'epoch' + str(epoch) + '/' + 'test_results/BtoA'):
                os.mkdir(root + 'epoch' + str(epoch) + '/' + 'test_results/BtoA')
            path = root + 'epoch' + str(epoch) + '/' + 'test_results/BtoA/' + str(iter) + '_input.png'
            plt.imsave(path, (realB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'test_results/BtoA/' + str(iter) + '_output.png'
            plt.imsave(path, (fakeA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'test_results/BtoA/' + str(iter) + '_recon.png'
            plt.imsave(path, (recB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

            

            util.show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
        del realA, realB, fakeA, fakeB, recA, recB
        torch.cuda.empty_cache()

    else:
        print("save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train")
        iter = 0
        for (img_A, _), (img_B, _) in zip(train_loader_A, train_loader_B):
            iter += 1
            if not os.path.isdir(root + 'epoch' + str(epoch) + '/' + 'train_results'):
                os.mkdir(root + 'epoch' + str(epoch) + '/' + 'train_results')
            if not os.path.isdir(root + 'epoch' + str(epoch) + '/' + 'train_results/makeup'):
                os.mkdir(root + 'epoch' + str(epoch) + '/' + 'train_results/makeup')
            
            if not os.path.isdir(root + 'epoch' + str(epoch) + '/' + 'train_results/AtoB'):
                os.mkdir(root + 'epoch' + str(epoch) + '/' + 'train_results/AtoB')
            realA = util.to_var(img_A, requires_grad=False)
            realB = util.to_var(img_B, requires_grad=False)
            middle_res = B_encoder(realA, realB)
            genB, genA = B_decoder(torch.cat((middle_res[1], middle_res[3]), dim=1))
            middle_res = B_encoder(genA, genB)
            recB, recA = B_decoder(torch.cat((middle_res[1], middle_res[3]), dim=1))
            # genB, genA = B_decoder(torch.cat((B_encoder(realA, realB)),dim=1))#G(realA, realB)
            # recB, recA = B_decoder(torch.cat((B_encoder(genA, genB)),dim=1))#G(genA, genB)
            path = root + 'epoch' + str(epoch) + '/' + 'train_results/makeup/' + str(iter) + '_inA.png'
            plt.imsave(path, (realA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'train_results/makeup/' + str(iter) + '_out.png'
            plt.imsave(path, (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'train_results/makeup/' + str(iter) + '_inB.png'
            plt.imsave(path, (realB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)


            path = root + 'epoch' + str(epoch) + '/' + 'train_results/AtoB/' + str(iter) + '_input.png'
            plt.imsave(path, (realA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'train_results/AtoB/' + str(iter) + '_output.png'
            plt.imsave(path, (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'train_results/AtoB/' + str(iter) + '_recon.png'
            plt.imsave(path, (recA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

            if not os.path.isdir(root + 'epoch' + str(epoch) + '/' + 'train_results/BtoA'):
                os.mkdir(root + 'epoch' + str(epoch) + '/' + 'train_results/BtoA')
            path = root + 'epoch' + str(epoch) + '/' + 'train_results/BtoA/' + str(iter) + '_input.png'
            plt.imsave(path, (realB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'train_results/BtoA/' + str(iter) + '_output.png'
            plt.imsave(path, (genA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            path = root + 'epoch' + str(epoch) + '/' + 'train_results/BtoA/' + str(iter) + '_recon.png'
            plt.imsave(path, (recB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            if iter > 9:
                break

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G_A.state_dict(),  root + model + 'generatorA_param.pkl')
torch.save(G_B.state_dict(),  root + model + 'generatorB_param.pkl')
torch.save(D_A.state_dict(),  root + model + 'discriminatorA_param.pkl')
torch.save(D_B.state_dict(),  root + model + 'discriminatorB_param.pkl')
util.show_train_hist(train_hist, save=True, path= root + model + 'train_hist.png')
