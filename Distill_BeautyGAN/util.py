# coding=utf-8
import itertools, imageio, torch, random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import datasets
from torch.autograd import Variable
import cv2
import torch
import torch.nn as nn
#from torch.utils.data.distributed import DistributedSampler

# torch.distributed.init_process_group(backend="nccl")
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
device = torch.device("cuda:0")#, local_rank)

def get_mask( input_face, detector, predictor, window=5):#detector, predictor,
    """
    get face, lip, eyes, eyebrows masks
    :param input_face: input image
    :param detector: A pretrained model for face detection
    :param predictor: A pretrained model generating 68 feature points of human face
    :param window: Get the window size of the eye shadow area around the eye
    :return: tuple(lip_mask, eye_mask, eyebrow_mask, face_mask)
    """
    gray = cv2.cvtColor(input_face, cv2.COLOR_RGB2GRAY)#BGRtoRGB
    dets = detector(gray, 1)

    for face in dets:
        shape = predictor(input_face, face)
        temp = []
        for pt in shape.parts():
            temp.append([pt.x, pt.y])
        lip_mask = np.zeros([256, 256])
        eye_mask = np.zeros([256, 256])
        eyebrow_mask = np.zeros([256, 256])
        face_mask = np.zeros([256, 256])

        # lip mask
        cv2.fillPoly(lip_mask, [np.array(temp[48:60]).reshape((-1, 1, 2))], (255, 255, 255))
        cv2.fillPoly(lip_mask, [np.array(temp[60:68]).reshape((-1, 1, 2))], (0, 0, 0))

        # left eye_mask
        left_left = min(x[0] for x in temp[36:42])
        left_right = max(x[0] for x in temp[36:42])
        left_bottom = min(x[1] for x in temp[36:42])
        left_top = max(x[1] for x in temp[36:42])
        left_rectangle = np.array(
            [[left_left - window, left_top + window], [left_right + window, left_top + window],
             [left_right + window, left_bottom - window], [left_left - window, left_bottom - window]]
        ).reshape((-1, 1, 2))
        cv2.fillPoly(eye_mask, [left_rectangle], (255, 255, 255))
        cv2.fillPoly(eye_mask, [np.array(temp[36:42]).reshape((-1, 1, 2))], (0, 0, 0))

        # right eye_mask
        right_left = min(x[0] for x in temp[42:48])
        right_right = max(x[0] for x in temp[42:48])
        right_bottom = min(x[1] for x in temp[42:48])
        right_top = max(x[1] for x in temp[42:48])
        right_rectangle = np.array(
            [[right_left - window, right_top + window], [right_right + window, right_top + window],
             [right_right + window, right_bottom - window], [right_left - window, right_bottom - window]]
        ).reshape((-1, 1, 2))
        cv2.fillPoly(eye_mask, [right_rectangle], (255, 255, 255))
        cv2.fillPoly(eye_mask, [np.array(temp[42:47]).reshape((-1, 1, 2))], (0, 0, 0))

        # face_mask and eyebrow_mask
        cv2.fillPoly(eyebrow_mask, [np.array(temp[17:22]).reshape(-1, 1, 2)], (255, 255, 255))
        cv2.fillPoly(eyebrow_mask, [np.array(temp[22:27]).reshape(-1, 1, 2)], (255, 255, 255))

        cv2.fillPoly(face_mask, [np.array(temp[0:27]).reshape(-1, 1, 2)], (255, 255, 255))

        return lip_mask, eye_mask, eyebrow_mask, face_mask

def to_var(x, requires_grad=True):
    x = x.type(torch.FloatTensor)
    if torch.cuda.is_available():
        x = x.to(device)
    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(x)

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def rebound_box(mask_A, mask_B, mask_A_face):
    index_tmp = mask_A.nonzero()
    x_A_index = index_tmp[:, 2]
    y_A_index = index_tmp[:, 3]
    index_tmp = mask_B.nonzero()
    x_B_index = index_tmp[:, 2]
    y_B_index = index_tmp[:, 3]
    mask_A_temp = mask_A.copy_(mask_A)
    mask_B_temp = mask_B.copy_(mask_B)
    mask_A_temp[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
                            mask_A_face[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
    mask_B_temp[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
                            mask_A_face[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
    mask_A_temp = to_var(mask_A_temp, requires_grad=False)
    mask_B_temp = to_var(mask_B_temp, requires_grad=False)
    return mask_A_temp, mask_B_temp

def mask_preprocess( mask_A, mask_B):
    index_tmp = mask_A.nonzero()
    x_A_index = index_tmp[:, 2]
    y_A_index = index_tmp[:, 3]
    index_tmp = mask_B.nonzero()
    x_B_index = index_tmp[:, 2]
    y_B_index = index_tmp[:, 3]
    mask_A = to_var(mask_A, requires_grad=False)
    mask_B = to_var(mask_B, requires_grad=False)
    index = [x_A_index, y_A_index, x_B_index, y_B_index]
    index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
    return mask_A, mask_B, index, index_2

def criterionHis( input_data, target_data, mask_src, mask_tar, index):

    L1_loss = nn.L1Loss().to(device)
    input_data = (de_norm(input_data) * 255).squeeze()
    target_data = (de_norm(target_data) * 255).squeeze()
    mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
    mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
    input_masked = input_data * mask_src
    target_masked = target_data * mask_tar
    input_match = histogram_matching(input_masked, target_masked, index)
    input_match = to_var(input_match, requires_grad=False)
    loss = L1_loss(input_masked, input_match)
    return loss

def histogram_loss_cal(source,template,source_mask,template_mask):
    """
    According to the given image template and its mask, corresponding histogram matching
    is carried out for the specific region of the original image source,
    and the difference of the original image before and after histogram matching is calculated

    :param source: the original image
    :param template: the target image
    :param source_mask:  the mask of the original image
    :param template_mask: the mask of the target image
    :return: MSE Loss of the original image before and after matching
        """
    source = source.clone().view( [1, -1])
    template = template.clone().view([1, -1])
    source_mask = source_mask.clone().view([-1, 256 * 256]).to(device)
    template_mask = template_mask.clone().view([-1, 256*256]).to(device)
    # According to the mask, all pixels outside a specific area are omitted

    source = source * source_mask
    template = template * template_mask
    source = source.cpu()
    template = template.cpu()
    # Obtain the histogram distribution of the original image and the target image on the pixel value
    his_bins = 255
    max_value = torch.max(torch.max(source), torch.max(template)).item()
    min_value = torch.min(torch.min(source), torch.min(template)).item()

    hist_delta = torch.div(torch.sub(max_value,min_value) , his_bins)
    hist_range = torch.range(min_value, max_value, hist_delta)

    hist_range = torch.add(hist_range, torch.div(hist_delta, 2))


    s_hist = torch.histc(source, his_bins, min_value, max_value)
    t_hist = torch.histc(template, his_bins, min_value, max_value)

    # The histogram distribution of the two is converted into the form of cumulative percentage
    s_quantiles = torch.cumsum(s_hist,0)
    s_last_element = torch.tensor(len(s_quantiles) - 1)
    s_quantiles = torch.div(s_quantiles, torch.gather(s_quantiles, 0, s_last_element))

    t_quantiles = torch.cumsum(t_hist,0)
    t_last_element = torch.tensor(len(t_quantiles) - 1)
    t_quantiles = torch.div(t_quantiles, torch.gather(t_quantiles, 0, t_last_element))

    # The target pixel value closest to each pixel value in the original image is obtained
    # according to the cumulative percentage of the target image
    nearest_indices = torch.tensor(list(map(lambda x: torch.argmin(torch.abs(torch.sub(t_quantiles, x))).item(),s_quantiles)))

    s_bin_index = torch.div(source, hist_delta).to(torch.int64)
    s_bin_index = torch.clamp(s_bin_index, 0, 254)

    tmp = torch.gather(nearest_indices, 0, s_bin_index[0])

    matched_to_t = torch.gather(hist_range, 0, tmp)

    # The original image and the matched image are standardized and the mean square error is calculated
    matched_to_t = torch.sub(torch.div(matched_to_t,127.5),1)
    source = torch.sub(torch.div(source,127.5),1)
    return torch.mean(torch.pow((matched_to_t-source),2))

def get_optimizer(model):

    optimizer = optim.Adam(model.parameters(), lr=0.002, betas=(0.5, 0.999))
    return optimizer


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['G_losses']))

    y3 = hist['G_A_losses']
    y4 = hist['G_B_losses']
    y5 = hist['A_cycle_losses']
    y6 = hist['B_cycle_losses']
    y7 = hist['histogram_losses']
    y8 = hist['perceptual_losses']
    y9 = hist['idt_losses']
    y10 = hist['G_losses']


    plt.plot(x, y3, label='G_A_loss')
    plt.plot(x, y4, label='G_B_loss')
    plt.plot(x, y5, label='A_cycle_loss')
    plt.plot(x, y6, label='B_cycle_loss')
    plt.plot(x, y7, label='histogram_loss')
    plt.plot(x, y8, label='perceptual_loss')
    plt.plot(x, y9, label='idt_loss')
    plt.plot(x, y10, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def generate_animation(root, model, opt):
    images = []
    for e in range(opt.train_epoch):
        img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(root + model + 'generate_animation.gif', images, fps=5)

def data_load(path, subfolder, transform, batch_size, shuffle=True):
    dset = datasets.ImageFolder(path, transform)   #shuffle 随机排序
    ind = dset.class_to_idx[subfolder]#0 2 3 1
    n = 0
    for i in range(dset.__len__()): #len:2661
        #print("ind=",ind,"n=", n, "dset.imgs[n][1]=", dset.imgs[n][1])
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)#, pin_memory=True, sampler=DistributedSampler(dset))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

class image_store():
    def __init__(self, store_size=50):
        self.store_size = store_size
        self.num_img = 0
        self.images = []

    def query(self, image):
        select_imgs = []
        for i in range(image.size()[0]):
            if self.num_img < self.store_size:
                self.images.append(image)
                select_imgs.append(image)
                self.num_img += 1
            else:
                prob = np.random.uniform(0, 1)
                if prob > 0.5:
                    ind = np.random.randint(0, self.store_size - 1)
                    select_imgs.append(self.images[ind])
                    self.images[ind] = image
                else:
                    select_imgs.append(image)

        return Variable(torch.cat(select_imgs, 0))

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images