from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024,get_unet_1024_LeakyReLu,get_unet_1024_PReLU,get_unet_1024_PReLU_Hans

input_size = 1024

max_epochs = 200
batch_size = 1

orig_width = 1918
orig_height = 1280

threshold = 0.5

#model_factory = get_unet_1024_LeakyReLu
model_factory = get_unet_1024_LeakyReLu
#model_factory = get_unet_1024_PReLU