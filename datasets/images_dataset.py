from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import PIL
import numpy as np
import torch
from torchvision import transforms
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

class ImagesDataset(Dataset):
    
	def __init__(self, source_root, target_root, face_source_root, face_target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.face_source_paths = sorted(data_utils.make_dataset(face_source_root))
		self.face_target_paths = sorted(data_utils.make_dataset(face_target_root))

		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		self.transform_mask = transforms.Compose([
        	transforms.Resize(256, interpolation=PIL.Image.NEAREST),
        	ToTensor])
	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		face_from_path = self.face_source_paths[index]
		face_from_im = Image.open(face_from_path)
		face_from_im = face_from_im.convert('RGB') if self.opts.label_nc == 0 else face_from_im.convert('L')

		from_im = Image.open(from_path)
		from_hm = self.transform_mask(from_im)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		
		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')

		face_to_path = self.face_target_paths[index]
		face_to_im = Image.open(face_to_path).convert('RGB')

		if self.target_transform:
			to_im = self.target_transform(to_im)
			face_to_im = self.target_transform(face_to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
			face_from_im = self.source_transform(face_from_im)
		else:
			from_im = to_im
			face_from_im = face_to_im
		return from_hm, from_im, to_im, face_from_im, face_to_im