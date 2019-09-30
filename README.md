## Setup
Tested with Python 3.6 via virtual environment:
```shell
$ python3.6 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Run on Abstract dataset
Use Abstract dataset(8145 images) as an example:

1. download abstract.zip from the author's link `https://drive.google.com/open?id=17Cm2352V9G1tR4kii5yHI_KUkevLC67_` and unzip into the `./data` folder
2. `python uniform.py` to convert the images to RGB and resize them
3. if you don't want to load checkpoints, make sure to empty checkpoint folder
4. `python main.py --train --crop`

the default parameter settings are specified in the following section in `main.py` file, which can be overwritten by providing values in the command line.

For example, the default epoch for training is 25 and dataset is in "abstract-resize" folder. To use a different value, you can do the following:

- first, download the flower dataset and unzip to ./data/flower
- revise the following paths in `uniform.py`

    ```python
    # path is the folder containing your data
    path = './data/flower/'

    # target_path is the folder storing resized images
    target_path = './data/flower-resize'
    ```
- run `python uniform.py` to generate resized images in `./data/flower-resize`

- train the model with flower dataset with 5 epochs: `python main.py --epoch=5 --dataset=flower-resize --train`, the checkpoints will be generated and saved in a subfolder in checkpoint folder with prefix "flower-resize"

<img width="390" alt="Screen Shot 2019-09-23 at 2 54 48 PM" src="https://user-images.githubusercontent.com/595772/65453941-2733e200-de12-11e9-9e7b-6f662b032d90.png">

```python
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 256, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "abstract-resize", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
flags.DEFINE_integer("option", 1, "visualization option [1]")
```

Pay attention to parameter `option`. Different option can generate different kinds of visualization, this is very useful for testing.
- option = 0: Cannot be used for MNIST. Generate a big map containing `batch_size` images generated randomly.
- option = 1: Almost do the same thing as OPTION 0, but can be applied for MNIST.
- option = 2: Generate a GIF lasting 2 seconds with `batch_size` frames. If failed, do the same thing as OPTION 1.
- option = 3: Cannot be used for MNIST, and generate the same GIF as OPTION 2.
- option = 4: Generate a big map containing `z_dim` GIFs.
So if you want to generate a big map containing 100 GIFs on flower dataset: 
```shell
python main.py --dataset flower-resize --option 4
```
## Training on CPU
- Training Equipment: Intel Xeon E3 4 cores
- Training Time: About 1 hour per epoch

## Parameters
- learning rate: 0.2
- beta1: 0.5
- batch_size: 64
- input size: 256 * 256
- output size: 128 * 128

## Result on Abstract.zip
- after 1 epoch
![](https://github.com/zhuojg/DCGAN-Art-Tensorflow/raw/master/samples/train_00_0099.png)

- after 10 epochs
![](https://github.com/zhuojg/DCGAN-Art-Tensorflow/raw/master/samples/train_10_0029.png)

- after 20 epochs
![](https://github.com/zhuojg/DCGAN-Art-Tensorflow/raw/master/samples/train_20_0059.png)

- Some test results after 25 epochs
![](https://github.com/zhuojg/DCGAN-Art-Tensorflow/raw/master/samples/test_arange_7.png)
![](https://github.com/zhuojg/DCGAN-Art-Tensorflow/raw/master/samples/test_arange_21.png)
![](https://github.com/zhuojg/DCGAN-Art-Tensorflow/raw/master/samples/test_arange_31.png)

## Train use Google Cloud Platform (GCP)

- make a copy of the author's GCP notebook: `https://colab.research.google.com/drive/18RglimpA1JH7bRbTXtxx9fAbDl60sFVQ#scrollTo=YLBwMdxMW3PR`

- make sure runtime is GPU:
    ![Screen Shot 2019-09-23 at 9 26 46 PM](https://user-images.githubusercontent.com/595772/65474289-37b37f00-de49-11e9-99fc-4d7a9efff487.png)
    ![Screen Shot 2019-09-23 at 9 27 03 PM](https://user-images.githubusercontent.com/595772/65474294-397d4280-de49-11e9-8561-f595bfe7a637.png)




# from the original author
# Art and Design DCGAN in Tensorflow

Modified version of Taehoon Kim’s tensorflow implementation of DCGAN `https://carpedm20.github.io/faces/` with a focus on generating paintings and soon graphic design.

It includes a script to scrape WikiArt, one to uniform images in a format that the Dcgan can work with and a Google Colab Notebook to train it on a free GPU.

![](https://pbs.twimg.com/media/DdvgUjdVwAAyANO.jpg:large)

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)

You can find a zip of my dataset in:
`https://drive.google.com/open?id=17Cm2352V9G1tR4kii5yHI_KUkevLC67_`

Checkpoints here:
`https://drive.google.com/open?id=1yABe4LsWeDQz5p5IO2AYJPosGOgqtD2Z`

Colab Notebook:
`https://colab.research.google.com/drive/18RglimpA1JH7bRbTXtxx9fAbDl60sFVQ#scrollTo=YLBwMdxMW3PR`

You will have to convert the images to RGB and resize them with: `uniform.py` by changing `path` with your directory

Train your own dataset:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --train

If your dataset is located in a different root directory:

    $ python main.py --dataset DATASET_NAME --data_dir DATASET_ROOT_DIR --train
    $ python main.py --dataset DATASET_NAME --data_dir DATASET_ROOT_DIR
    $ # example
    $ python main.py --dataset=eyes --data_dir ../datasets/ --input_fname_pattern="*_cropped.png" --train
