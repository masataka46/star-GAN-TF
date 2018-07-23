import numpy as np
import os
from PIL import Image


def convert_to_10class(d):
    d_mod = np.zeros((len(d), 10), dtype=np.float32)
    for num, contents in enumerate(d):
        d_mod[num][int(contents)] = 1.0
    # debug
    # print("d_mod[100] =", d_mod[100])
    # print("d_mod[200] =", d_mod[200])

    return d_mod

def make_1_img(img_batch):  # for debug
    for num, ele in enumerate(img_batch):
        if num != 0:
            continue

        img_tmp = ele
        img_tmp = np.tile(img_tmp, (1, 1, 3)) * 255
        img_tmp = img_tmp.astype(np.uint8)
        image_PIL = Image.fromarray(img_tmp)
        image_PIL.save("./out_images_tripleGAN/debug_img_" + ".png")

    return

def unnorm_img(img_np):
    img_np_255 = (img_np + 1.0) * 127.5
    img_np_255_mod1 = np.maximum(img_np_255, 0)
    img_np_255_mod1 = np.minimum(img_np_255_mod1, 255)
    img_np_uint8 = img_np_255_mod1.astype(np.uint8)
    return img_np_uint8


def convert_np2pil(images_255):
    list_images_PIL = []
    for num, images_255_1 in enumerate(images_255):
        image_1_PIL = Image.fromarray(images_255_1)
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL
    
def make_output_img(ori_img, gen_img, gen_gen_img, sample_num_h, out_image_dir, epoch, log_file_name):

    img1_h = len(ori_img[1])
    img1_w = len(ori_img[2])
    # print("img1_h, ", img1_h)
    # print("img1_w, ", img1_w)

    ori_img_255 = unnorm_img(ori_img)
    list_ori_img_PIL = convert_np2pil(ori_img_255)
    gen_img_255 = unnorm_img(gen_img)
    list_gen_img_PIL = convert_np2pil(gen_img_255)
    gen_gen_img_255 = unnorm_img(gen_gen_img)
    list_gen_gen_img_PIL = convert_np2pil(gen_gen_img_255)

    wide_image_np = np.zeros((sample_num_h * img1_h, img1_w * 9, 3), dtype=np.uint8)
    wide_image_PIL = Image.fromarray(wide_image_np)
    for num, image_PIL_1 in enumerate(list_ori_img_PIL):
        wide_image_PIL.paste(image_PIL_1, ((num // sample_num_h) * img1_w * 3, (num % sample_num_h) * img1_h))
    for num, image_PIL_1 in enumerate(list_gen_img_PIL):
        wide_image_PIL.paste(image_PIL_1, ((num // sample_num_h) * img1_w * 3 + img1_w, (num % sample_num_h) * img1_h))
    for num, image_PIL_1 in enumerate(list_gen_gen_img_PIL):
        wide_image_PIL.paste(image_PIL_1, ((num // sample_num_h) * img1_w * 3 + 2 * img1_w, (num % sample_num_h) * img1_h))

    wide_image_PIL.save(out_image_dir + "/resultImage_"+ log_file_name + '_' + str(epoch) + ".png")

    return




