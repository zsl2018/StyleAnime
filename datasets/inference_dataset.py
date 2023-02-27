from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import torch
from torchvision import transforms
import PIL
import numpy as np

class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im

def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img
		
class InferenceDataset_enc(Dataset):

	def __init__(self, root_1, root_2, transform_1=None, transform_2=None, opts=None):
		self.paths_1 = sorted(data_utils.make_dataset(root_1))
		self.paths_2 = sorted(data_utils.make_dataset(root_2))
		self.transform_1 = transform_1
		self.transform_2 = transform_2

		self.opts = opts

	def __len__(self):
		return len(self.paths_1)

	def __getitem__(self, index):
		from_path = self.paths_1[index]
		from_im = Image.open(from_path)

		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.paths_2[index]
		to_im = Image.open(to_path).convert('RGB')

		if self.transform_1:
			from_im = self.transform_1(from_im)
		if self.transform_2:
			to_im = self.transform_2(to_im)
		return from_im, to_im





