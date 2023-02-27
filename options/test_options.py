from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--parsing_data_path', type=str, default='/mnt/lab/lzs/dolphin_copy/psp_220926/results/portrait2anime/seg', help='Path to directory of input parsing data')
		self.parser.add_argument('--image_data_path', type=str, default='/mnt/lab/lzs/dolphin_copy/psp_220926/results/portrait2anime/img', help='Path to directory of input reference image data')
		self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')
		self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')

		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
		self.parser.add_argument('--alpha', default=0.1, type=float, help='alpha value in FaceBank aggregation')
		self.parser.add_argument('--bank_list', type=str, default='./pretrained_models/latent_bank_list.npy', help='alpha value in FaceBank aggregation')

		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')
		self.parser.add_argument('--resize_factors', type=str, default=None,
		                         help='Downsampling factor for super-res (should be a single value for inference).')

	def parse(self):
		opts = self.parser.parse_args()
		return opts