from multiprocessing import Pool, cpu_count
from glob import glob
from matplotlib import pyplot as plt
import cv2
output_size = 128 # 64
dir_imgs = "edges2shoes"
fn_train_imgs = glob(f"{dir_imgs}/train/*.*")
len(fn_train_imgs)

edges_save_path = "edges2shoes/train/edges/"
shoes_save_path = "edges2shoes/train/shoes/"

def split_image(fn):
    raw_fn = fn.split("/")[-1]
    im = plt.imread(fn)[..., :3]

    edges_im = im[:,:256,:]
    shoes_im = im[:,256:,:]

    if output_size == 128:
        edges_im = cv2.resize(edges_im, (output_size,output_size))
        shoes_im = cv2.resize(shoes_im, (output_size,output_size))
    elif output_size == 64:
        edges_im = cv2.resize(cv2.erode(edges_im, np.ones((2,2),np.uint8), iterations = 1), (output_size,output_size))
        shoes_im = cv2.resize(shoes_im, (output_size,output_size))
    else:
        assert (output_size == 128) or (output_size == 64), "output_size should be either 128 or 64."

    plt.imsave(f"{edges_save_path}{raw_fn}", edges_im, format="jpg")
    plt.imsave(f"{shoes_save_path}{raw_fn}", shoes_im, format="jpg")

    return None

from joblib import Parallel, delayed
_ = Parallel(n_jobs=-1, verbose=1)(map(delayed(split_image), fn_train_imgs))
print(len(glob("edges2shoes/train/edges/*.*")))
assert len(glob("edges2shoes/train/edges/*.*")) == len(glob("edges2shoes/train/shoes/*.*"))
