import glob
from matplotlib import pyplot as plt
import numpy as np
import os

def extract_images(path):
    """
    this function is the first step in turning a directory into a
    dataset. If you have raw data, start here.
    """
    images = []
    for furniture in ["Bed/*", "Chair/*", "Sofa/*"]:
        p = os.path.join(path, furniture)
        for file in glob.glob(p):
            filename = file.split("/")[-1]
            if "bed" in filename.lower(): # get classifier index
                idx = 0
            elif "chair" in filename.lower():
                idx = 1
            elif "sofa" in filename.lower():
                idx = 2
            else:
                raise Exception("All filenames must have a class in them.")
            img = plt.imread(file)
            img = img[:,:,:3] # remove alpha channels, if any
            images.append((img, idx))
    return images
