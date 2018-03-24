import numpy as np
import struct

def read(img_path, lbl_path):
    """
    Python function for importing the MNIST data set.
    It returns 2-tuples of list with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    with open(lbl_path, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(img_path, 'rb') as fimg:
        _, num, _, _ = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(num, -1)

    return (img, lbl)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap='gray')
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

if __name__ == '__main__':
    img, lbl = read('./train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    print(lbl[3])
    show(img[3])
