# Star-GAN-Tensorflow  
Tensorflow implementation of [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)  

## Issue  
I am now modifying the Wasserstein Loss to Least square Loss.

## Usage  
```bash
> python train_starGAN.py --epoch 400 --log_file_name LOG_FILE_NAME --batch_size 16 --dir_name1 DOMAIN1_DIRECTORY_NAME --dir_name2 DOMAIN2_DIRECTORY_NAME --dir_name3 DOMAIN3_DIRECTORY_NAME
```

## Result image  
after 200 epochs, the result image is below.  
![resultimage_180722_1070_cat_04_210](https://user-images.githubusercontent.com/15444879/43061919-0cdc8702-8e92-11e8-9729-b9232f778924.png)  
The task is converting siamese cat, tiger cat, sbyssinian cat to another cat.  