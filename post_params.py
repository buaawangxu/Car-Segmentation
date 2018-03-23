from model.post_Models import get_post_process_inception_res_sen_dense
input_size = 256

max_epochs = 200
batch_size = 1

orig_width = 1918
orig_height = 1280

threshold = 0.5

#model_factory = get_unet_1024_LeakyReLu
model_factory = get_post_process_inception_res_sen_dense(input_shape = (input_size,input_size,4), pool_size=(2, 2), n_calsses=1, n_base_filters=4)

#model_factory = get_unet_1024_PReLU
