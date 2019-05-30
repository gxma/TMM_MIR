function [rcnn_model, mdf] = init_deepmodel()

mdf.rcnn_model_file = './models/ilsvrc.mat';   %get the imageNet deep learning model

mdf.model_dir = './models/bvlc_alexnet/';
mdf.net_model = [mdf.model_dir 'deploy.prototxt'];
mdf.net_weights = [mdf.model_dir 'bvlc_alexnet.caffemodel'];
mdf.phase = 'test';
mdf.modelname = 'alex_caffe';
mdf.batchsize = 30;
use_gpu = 1;
rcnn_model = rcnn_load_model(mdf.rcnn_model_file, use_gpu);