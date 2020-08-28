## checkpoint directory (under checkpoint_dir) must be the same name as dataset, plus _batchsize_outputw_output_h
## for instance, if dataset is insects-square-512-rgb, data_dir is ./data, checkpoint_dir is ./checkpoints,
## checkpoints must be in ./checkpoints/insects-square-512-rgb_6_512_512/
## and the imagess must be in ./data/insects-square-512-rgb . but they don't need to be the same images, just same depth and dimensions
## generate_test_images sometimes shows up as an error:
## "tensorflow.python.framework.errors_impl.InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [100,2097152] rhs shape= [10,2097152]
## use the last shape. 10 in the example above

## example command

python generate_image.py --input_width=512 --input_height=512 --output_width=512 --output_height=512 --batch_size=16 --sample_num=12 --z_dim=12 --dataset=insects_new --checkpoint_dir=/home/art/apps/machinelearning/tensorflow/dcgan/DCGAN-Art-Tensorflow/checkpoint/ --data_dir=/home/art/apps/datasets/ --generate_test_images=12
