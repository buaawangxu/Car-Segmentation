from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024,get_unet_1024_LeakyReLu,get_unet_1024_PReLU,get_unet_1024_PReLU_Hans
from model.Models import get_unet
input_size = 256

max_epochs = 200
batch_size = 1

orig_width = 1918
orig_height = 1280

threshold = 0.5

#model_factory = get_unet_1024_LeakyReLu
model_factory = get_unet(input_shape = (input_size,input_size,3), pool_size=(2, 2), n_calsses=1, depth=3, n_base_filters=8)
	#get_unet_FPN(input_shape = (input_size,input_size,3), pool_size=(2, 2), n_calsses=1, depth=3, n_base_filters=8)

#model_factory = get_unet_1024_PReLU
