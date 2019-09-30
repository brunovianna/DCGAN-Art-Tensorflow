from PIL import Image
import os, sys

# path is the folder containing your data
path = './data/abstract/'

# target_path is the folder storing resized images
target_path = './data/abstract-resize'

target_size = (256, 256)

dirs = os.listdir(path)
if not os.path.isdir(target_path):
    os.mkdir(target_path)

for item in dirs:
    try:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            longer_side = max(im.size)

            horizontal_padding = (longer_side - im.size[0]) / 2
            vertical_padding = (longer_side - im.size[1]) / 2
            f, e = os.path.splitext(path+item)
            imResize = im.crop(
            (
                -horizontal_padding,
                -vertical_padding,
                im.size[0] + horizontal_padding,
                im.size[1] + vertical_padding
            )
            )
            RGB = imResize.convert('RGB')
            little = RGB.resize((256, 256), Image.ANTIALIAS)

            little.save(target_path + '/' + item, 'JPEG', quality=50)
            # little.save(f + '_resized.jpg', 'JPEG', quality=30)
            print("saving {}".format(f))

    except Exception as e:
        print(e)
