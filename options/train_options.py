from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--dataset_type', default='anime_seg_to_face', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')

		self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=1, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=1, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--loss_adv_weight', default=0.1, type=float, help='Image adversarial loss weight')
		self.parser.add_argument('--loss_adv_weight_latent', default=0.1, type=float, help='Latent code adversarial loss weight')

		self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='The learning rate of Encoders')
		self.parser.add_argument('--learning_rate_d', default=0.0001, type=float, help='The learning rate of image Discriminator')
		self.parser.add_argument('--learning_rate_d_latent', default=0.0001, type=float, help='The learning rate of latent code Discriminator')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
		self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--start_from_latent_avg', action='store_true',
		                         help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space insteaf of w+')

		self.parser.add_argument('--hm_lambda', default=0.1, type=float, help='Color-preservation Loss, histogram match to preserve color')
		# self.parser.add_argument('--adv_lambda_two', default=1, type=float, help='Adverisarial loos multiplier factor in inner')
		self.parser.add_argument('--lambda_his_face', default=0.01, type=float, help='lambda_his_face loss multiplier factor')
		self.parser.add_argument('--lambda_his_hair', default=0.01, type=float, help='lambda_his_hair loss multiplier factor')
		self.parser.add_argument('--lambda_his_eye', default=0.01, type=float, help='lambda_his_hair loss multiplier factor')
		
		self.parser.add_argument('--lpips_lambda', default=2, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=2.5, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
		self.parser.add_argument('--w_norm_lambda_1', default=0, type=float, help='W-norm loss multiplier factor')
		self.parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
		self.parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')

		self.parser.add_argument('--stylegan_weights', default=model_paths['anime_ffhq'], type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--checkpoint_path', type=str, help='Path to pSp model checkpoint')

		self.parser.add_argument('--max_steps', default=300000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

		self.parser.add_argument('--resize_factors', type=str, default=None,
		                         help='For super-res, comma-separated resize factors to use for inference.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts