import numpy as np
from PIL import Image
import utility as Utility
import os
import random

class Make_datasets_many():

    def __init__(self, dir_name1, dir_name2, dir_name3, img_width, img_height, seed):
        self.dir_name1 = dir_name1
        self.dir_name2 = dir_name2
        self.dir_name3 = dir_name3
        self.img_width = img_width
        self.img_height = img_height
        self.seed = seed
        self.DOMAIN_NUM = 3
        self.file_list1 = self.get_file_names(self.dir_name1)
        self.data_num1 = len(self.file_list1)
        self.file_list2 = self.get_file_names(self.dir_name2)
        self.data_num2 = len(self.file_list2)
        self.file_list3 = self.get_file_names(self.dir_name3)
        self.data_num3 = len(self.file_list3)

        print("self.data_num", self.data_num1)
        print("self.file_list[0], ", self.file_list1[0])
        print("self.file_list[:10], ", self.file_list1[:10])
        print("self.data_num2", self.data_num2)
        print("self.file_list2[0], ", self.file_list2[0])
        print("self.file_list2[:10], ", self.file_list2[:10])
        print("self.data_num3", self.data_num3)
        print("self.file_list3[0], ", self.file_list3[0])
        print("self.file_list3[:10], ", self.file_list3[:10])
        random.seed(self.seed)

        # self.initial_noise = self.make_random_z_with_norm(mean, stddev, data_num, unit_num)
        # self.initial_file_list = self.file_list1[:4] + self.file_list2[:4]

    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files


    def read_data(self, filename_list, width, height):
        images = []
        for num, filename in enumerate(filename_list):

            pilIn = Image.open(filename)
            pilIn_size = pilIn.size #(width, height)

            if pilIn_size[0] >= pilIn_size[1]:
                margin_w = pilIn_size[0] - pilIn_size[1]
                margin_left = random.randint(0, margin_w)
                pil_crop = pilIn.crop((margin_left, 0, margin_left + pilIn_size[1], pilIn_size[1]))
                # print("pil_crop.size(case 1), ", pil_crop.size)
                pil_Resize = pil_crop.resize((width, height))
            else:
                margin_h = pilIn_size[1] - pilIn_size[0]
                margin_upper = random.randint(0, margin_h)
                pil_crop = pilIn.crop((0, margin_upper, pilIn_size[0], margin_upper + pilIn_size[0]))
                # print("pil_crop.size(case 2), ", pil_crop.size)
                pil_Resize = pil_crop.resize((width, height))

            image = np.asarray(pil_Resize, dtype=np.float32)
            #for mono-color image
            if image.ndim == 2:
                image = image.reshape((image.shape[0], image.shape[1], 1))
                image = np.tile(image, (1, 1, 3))

            images.append(image)

        return np.asarray(images)


    def normalize_data(self, data):
        data0_2 = data / 127.5
        data_norm = data0_2 - 1.0

        return data_norm


    def make_one_hot_channel(self, images, channel_num, one_value_channel_num):
        class_channel = np.zeros((images.shape[0], images.shape[1], images.shape[2], channel_num), dtype=np.float32)
        class_channel[:,:,:,one_value_channel_num] = 1.0
        return class_channel


    def make_one_hot_channel_for_fake(self, images, channel_num, not_channel_num_list):
        class_channel = np.zeros((images.shape[0], images.shape[1], images.shape[2], channel_num), dtype=np.float32)
        # print("class_channel.shape, ", class_channel.shape)
        half_num = images.shape[0] // 2
        # print("half_num, ", half_num)
        class_channel[:half_num,:,:,not_channel_num_list[0]] = 1.0
        class_channel[half_num:,:,:,not_channel_num_list[1]] = 1.0

        return class_channel


    def make_data_for_1_epoch(self):
        self.filename1_1_epoch = random.sample(self.file_list1, self.data_num1)
        self.filename2_1_epoch = random.sample(self.file_list2, self.data_num2)
        self.filename3_1_epoch = random.sample(self.file_list3, self.data_num3)

        return min(len(self.filename1_1_epoch),  len(self.filename2_1_epoch), len(self.filename3_1_epoch))


    def get_data_for_1_batch(self, i, batchsize):
        self.filename1_batch = self.filename1_1_epoch[i:i + batchsize]
        self.filename2_batch = self.filename2_1_epoch[i:i + batchsize]
        self.filename3_batch = self.filename3_1_epoch[i:i + batchsize]

        # make images
        images1 = self.read_data(self.filename1_batch, self.img_width, self.img_height)
        self.images1_n = self.normalize_data(images1)
        images2 = self.read_data(self.filename2_batch, self.img_width, self.img_height)
        self.images2_n = self.normalize_data(images2)
        images3 = self.read_data(self.filename3_batch, self.img_width, self.img_height)
        self.images3_n = self.normalize_data(images3)
        images_con = np.concatenate((self.images1_n, self.images2_n, self.images3_n), axis=0)

        # make one-hot channel label for fake image
        class_channel1_f = self.make_one_hot_channel_for_fake(self.images1_n, self.DOMAIN_NUM, [1, 2])
        class_channel2_f = self.make_one_hot_channel_for_fake(self.images2_n, self.DOMAIN_NUM, [0, 2])
        class_channel3_f = self.make_one_hot_channel_for_fake(self.images3_n, self.DOMAIN_NUM, [0, 1])
        class_channel_all_f = np.concatenate((class_channel1_f, class_channel2_f, class_channel3_f), axis=0)

        # make one-hot channel label for reconstructed image
        class_channel1_r = self.make_one_hot_channel(self.images1_n, self.DOMAIN_NUM, 0)
        class_channel2_r = self.make_one_hot_channel(self.images2_n, self.DOMAIN_NUM, 1)
        class_channel3_r = self.make_one_hot_channel(self.images3_n, self.DOMAIN_NUM, 2)
        class_channel_all_r = np.concatenate((class_channel1_r, class_channel2_r, class_channel3_r), axis=0)

        return images_con, class_channel_all_f, class_channel_all_r


    def get_test_data_for_1_batch(self, i, batchsize):
        self.filename1_batch = self.file_list1[i:i + batchsize]
        self.filename2_batch = self.file_list2[i:i + batchsize]
        self.filename3_batch = self.file_list3[i:i + batchsize]

        images1 = self.read_data(self.filename1_batch, self.img_width, self.img_height)
        self.images1_n = self.normalize_data(images1)
        images2 = self.read_data(self.filename2_batch, self.img_width, self.img_height)
        self.images2_n = self.normalize_data(images2)
        images3 = self.read_data(self.filename3_batch, self.img_width, self.img_height)
        self.images3_n = self.normalize_data(images3)
        images_con = np.concatenate((self.images1_n, self.images2_n, self.images3_n), axis=0)

        #make one-hot channel label for fake image
        class_channel1_f = self.make_one_hot_channel_for_fake(self.images1_n, self.DOMAIN_NUM, [1, 2])
        class_channel2_f = self.make_one_hot_channel_for_fake(self.images2_n, self.DOMAIN_NUM, [0, 2])
        class_channel3_f = self.make_one_hot_channel_for_fake(self.images3_n, self.DOMAIN_NUM, [0, 1])
        class_channel_all_f = np.concatenate((class_channel1_f, class_channel2_f, class_channel3_f), axis=0)

        # make one-hot channel label for reconstructed image
        class_channel1_r = self.make_one_hot_channel(self.images1_n, self.DOMAIN_NUM, 0)
        class_channel2_r = self.make_one_hot_channel(self.images2_n, self.DOMAIN_NUM, 1)
        class_channel3_r = self.make_one_hot_channel(self.images3_n, self.DOMAIN_NUM, 2)
        class_channel_all_r = np.concatenate((class_channel1_r, class_channel2_r, class_channel3_r), axis=0)

        return images_con, class_channel_all_f, class_channel_all_r


    def make_random_z_with_norm(self, mean, stddev, data_num, unit_num):
        return np.random.normal(mean, stddev, (data_num, unit_num))


    def make_target_1_0(self, value, data_num):
        if value == 0.0:
            target = np.zeros((data_num, 1), dtype=np.float32)
        elif value == 1.0:
            target = np.ones((data_num, 1), dtype=np.float32)
        else:
            print("target value error")

        return target


    def make_target_for_DC_loss(self):
        data_num = len(self.images1_n) + len(self.images2_n)
        target_f = np.zeros((data_num, self.DOMAIN_NUM), dtype=np.float32)
        target_f[:len(self.images1_n), 1] = 1.0
        target_f[len(self.images1_n):, 0] = 1.0

        target_r = np.zeros((data_num, self.DOMAIN_NUM), dtype=np.float32)
        target_r[:len(self.images1_n), 0] = 1.0
        target_r[len(self.images1_n):, 1] = 1.0
        
        return target_f, target_r

    def make_target_for_DC_loss_from_channel(self, class_channel_all_f, class_channel_all_r):
        target_f = class_channel_all_f[:, 0, 0, :]
        target_r = class_channel_all_r[:, 0, 0, :]

        return target_f, target_r
    

if __name__ == '__main__':
    #debug
    dir_name = '/media/webfarmer/HDCZ-UT/dataset/food101/food-101/images/hot_dog/'
    img_width = 64
    img_height = 64
    '''
    make_datasets_food101 = Make_datasets_food101(dir_name, img_width, img_height)
    num = make_datasets_food101.make_data_for_1_epoch()
    
    filename_1_epoch = make_datasets_food101.filename_1_epoch
    print("filename_1_epoch[:10], ", filename_1_epoch[:10])
    images_n = make_datasets_food101.get_data_for_1_batch(4, 3)
    '''