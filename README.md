## Setup

Download data from the author's link and put in data folder

Tested with Python 3.6 via virtual environment:
```shell
$ python3.6 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Run

Use Abstract dataset as an example:

1. download abstract.zip and unzip into the /data folder
2. `python3 uniform.py` to convert the images to RGB and resize them
3. `python main.py --dataset=abstract --input_fname_pattern="*_resized.png" --train`


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
