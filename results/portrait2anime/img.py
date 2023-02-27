import glob
from PIL import Image

path1 = "./img/"
files = sorted(glob.glob(path1+"*.jpg"))

for i in range(len(files)):
    img = Image.open(files[i])
    img = img.resize((256, 256), Image.ANTIALIAS)
    img.save(files[i])