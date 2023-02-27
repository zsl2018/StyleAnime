import sys


import os
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset, InferenceDataset_enc
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
#from models.styleAnime import StyleAnime
from models.styleAnime import StyleAnime


# from utils.manifold import get_inter

def run():
	test_opts = TestOptions().parse()
	if test_opts.resize_factors is not None:
		assert len(test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
		                                'downsampling_{}'.format(test_opts.resize_factors))
		out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
										'downsampling_{}'.format(test_opts.resize_factors))
	else:
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
		out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

	os.makedirs(out_path_results, exist_ok=True)
	os.makedirs(out_path_coupled, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	opts = Namespace(**opts)
	net = StyleAnime(opts)

	net.eval()
	net.cuda()

	print('Loading dataset for {}'.format(opts.dataset_type))
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	dataset = InferenceDataset_enc(root_1 = opts.parsing_data_path,
								root_2 = opts.image_data_path,
	                           transform_1=transforms_dict['transform_inference'],
							   transform_2 = transforms_dict['transform_test'],
	                           opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=True)

	if opts.n_images is None:
		opts.n_images = len(dataset)
	
	global_i = 0
	global_time = []

	for input_batch in tqdm(dataloader):
		#print("input_batch",input_batch.shape)
		x, y = input_batch
		if global_i >= opts.n_images:
			break
		with torch.no_grad():
			x = x.cuda().float()
			y = y.cuda().float()
			tic = time.time()
			result_batch = run_on_batch(x, y, net, opts)
			toc = time.time()
			global_time.append(toc - tic)

		for i in range(opts.test_batch_size):
			result = tensor2im(result_batch[i])
			im_path = dataset.paths_1[global_i]

			if opts.couple_outputs or global_i % 100 == 0:
				input_im = log_input_image(x[i], opts)
				resize_amount = (256, 256)# if opts.resize_outputs else (1024, 1024)
				if opts.resize_factors is not None:
					source = Image.open(im_path)
					res = np.concatenate([np.array(source.resize(resize_amount)),
										  np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
										  np.array(result.resize(resize_amount))], axis=1)
				else:
					# otherwise, save the original and output
					res = np.concatenate([np.array(input_im.resize(resize_amount)),
										  np.array(result.resize(resize_amount))], axis=1)
				Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

			im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
			Image.fromarray(np.array(result)).save(im_save_path)

			global_i += 1

	stats_path = os.path.join(opts.exp_dir, 'stats.txt')
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)

	with open(stats_path, 'w') as f:
		f.write(result_str)


def run_on_batch(x, y, net, opts):
	if opts.alpha>0:
		from utils.manifold import get_inter

		_, latents = net(x,y, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
		latents_gen = get_inter(nearnN=10, w_c=opts.alpha, generated_f=latents, feature_list=np.load(opts.bank_list), dtype=np.float32)
		_, result_batch = net(latents_gen, y, return_latents=False, ref=True)
	else:
		result_batch, latents = net(x,y, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
		# torch.save(latents, 'latent_recon_01.pt')
	
	return result_batch

if __name__ == '__main__':
	run()