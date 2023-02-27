import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, w_norm
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.styleAnime import StyleAnime

from training.ranger import Ranger
from hm import hm_loss
from models.stylegan2.model import Discriminator, Discriminator_light
import torch.autograd as autograd
from models.discriminator_latent import LatentCodesDiscriminator
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.sum(torch.abs(in0 - in1), dim=1, keepdim=True)

class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		# Initialize network
		self.net = StyleAnime(self.opts).to(self.device)
		self.dis = Discriminator(size=256).to(self.device)
		#self.dis = Discriminator_light().to(self.device)
		self.dis_latent = LatentCodesDiscriminator(style_dim=512, n_mlp=4).to(self.device)
		# Initialize loss
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)

		if self.opts.hm_lambda > 0:
			self.hm_loss = hm_loss.HMLoss(lambda_his_face=opts.lambda_his_face, lambda_his_hair=opts.lambda_his_hair, lambda_his_eye=opts.lambda_his_eye)

		self.mse_loss = nn.MSELoss().to(self.device).eval()


		# Initialize optimizer
		self.optimizer, self.optimizer_d, self.optimizer_latent = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				
				x_hm, x, y, x_1, y_1 = batch
				x_hm, x, y, x_1, y_1 = x_hm.to(self.device).float(), x.to(self.device).float(), y.to(self.device).float(), x_1.to(self.device).float(), y_1.to(self.device).float()
				
				y_hat, latent = self.net.forward(x,y, return_latents=True)
				y_hat_1, latent_1 = self.net.forward(x_1, y_1, return_latents=True)


				if self.opts.loss_adv_weight_latent > 0: 
					code_real = self.dis_latent(latent)
					code_fake = self.dis_latent(latent_1)

					code_loss = self.discriminator_latent_loss(code_real, code_fake)
					self.optimizer_latent.zero_grad()
					code_loss.backward()
					self.optimizer_latent.step()

				if self.opts.loss_adv_weight > 0:
					y_real = self.dis(y)
					y_fake = self.dis(y_hat.detach())
					y_fake_1 = self.dis(y_hat_1.detach())

					loss_real = self.GAN_loss(y_real, real=True)
					loss_fake = self.GAN_loss(y_fake, real=False)
					loss_fake_1 = self.GAN_loss(y_fake_1, real=False)
					loss_gp = self.div_loss_(y, y_hat.detach(), cuda=self.device)

					d_loss = loss_real + loss_fake + 5 * loss_gp + loss_fake_1
					#d_loss = loss_real + loss_fake + 5 * loss_gp + loss_fake_1*self.opts.adv_lambda_two
					self.optimizer_d.zero_grad()
					d_loss.backward()
					self.optimizer_d.step()
				

				y_hat, latent = self.net.forward(x,y, return_latents=True)
				y_hat_1, latent_1 = self.net.forward(x_1, y_1, return_latents=True)

				loss, loss_dict, id_logs = self.calc_loss(x_hm, x, y, y_hat, latent, y_hat_1, latent_1)

				self.optimizer.zero_grad()
				loss.backward()
				#torch.nn.utils.clip_grad_norm_(self.net.encoder.parameters(), 1)
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x_hm, x, y, x_1, y_1 = batch

			with torch.no_grad():
				x_hm, x, y, x_1, y_1 = x_hm.to(self.device).float(), x.to(self.device).float(), y.to(self.device).float(), x_1.to(self.device).float(), y_1.to(self.device).float()
				y_hat, latent = self.net.forward(x, y, return_latents=True)
				y_hat_1, latent_1 = self.net.forward(x_1, y_1, return_latents=True)


				loss, cur_loss_dict, id_logs = self.calc_loss(x_hm, x, y, y_hat, latent, y_hat_1, latent_1)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hat,
									  title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		params_d = list(self.dis.parameters())
		optimizer_d = torch.optim.Adam(params_d, lr=self.opts.learning_rate_d)

		params_latent = list(self.dis_latent.parameters())
		optimizer_latent = torch.optim.Adam(params_latent, lr=self.opts.learning_rate_d_latent)
		return optimizer, optimizer_d, optimizer_latent

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
		print('Loading dataset for {}'.format(self.opts.dataset_type))
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset_celeba = ImagesDataset(source_root=dataset_args['train_source_root'],
		                                     target_root=dataset_args['train_target_root'],
											 face_source_root=dataset_args['face_train_source_root'],
											 face_target_root=dataset_args['face_train_target_root'],
		                                     source_transform=transforms_dict['transform_source'],
		                                     target_transform=transforms_dict['transform_gt_train'],
		                                     opts=self.opts)
		test_dataset_celeba = ImagesDataset(source_root=dataset_args['test_source_root'],
		                                    target_root=dataset_args['test_target_root'],
											face_source_root=dataset_args['face_test_source_root'],
											face_target_root=dataset_args['face_test_target_root'],
		                                    source_transform=transforms_dict['transform_source'],
		                                    target_transform=transforms_dict['transform_test'],
		                                    opts=self.opts)
		train_dataset = train_dataset_celeba
		test_dataset = test_dataset_celeba
		print("Number of training samples: {}".format(len(train_dataset)))
		print("Number of test samples: {}".format(len(test_dataset)))
		return train_dataset, test_dataset

	def calc_loss(self, x_hm, x, y, y_hat, latent, y_hat_1, latent_1):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.w_norm_lambda_1 > 0:
			loss_w_norm_1 = self.w_norm_loss(latent_1, self.net.latent_avg)
			loss_dict['loss_w_norm_1'] = float(loss_w_norm_1)
			loss += loss_w_norm_1 * self.opts.w_norm_lambda_1
		if self.opts.hm_lambda > 0:
			loss_hm = self.hm_loss(y_hat, y, x_hm, x_hm)
			loss_dict['loss_hm'] = float(loss_hm)
			loss += loss_hm * self.opts.hm_lambda
		if self.opts.loss_adv_weight > 0:
			y_adv = self.dis(y_hat)
			loss_adv = self.GAN_loss(y_adv, real=True)
			loss_dict['loss_adv'] = float(loss_adv)
			loss += loss_adv * self.opts.loss_adv_weight
		if self.opts.loss_adv_weight_latent > 0:
			y_adv_latent = self.dis_latent(latent_1)
			loss_adv_latent = F.softplus(-y_adv_latent).mean()
			loss_dict['loss_adv_latent'] = float(loss_adv_latent)
			loss += loss_adv_latent * self.opts.loss_adv_weight_latent

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=1):
		
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], self.opts),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict
	
	def GAN_loss(self, scores_out, real=True):
		if real:
			return torch.mean(F.softplus(-scores_out))
		else:
			return torch.mean(F.softplus(scores_out))

	def div_loss_(self, real_x, fake_x, p=2, cuda=False):
    # if cuda:
		x_ = real_x.requires_grad_(True)
		y_ = self.dis(x_)
		# cal f'(x)
		grad = autograd.grad(
			outputs=y_,
			inputs=x_,
			grad_outputs=torch.ones_like(y_),
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
			)[0]
		# grad = grad.view(x_.shape[0], -1)
		# div = (grad.norm(2, dim=1) ** p).mean()
		div = (grad * grad).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
		div = torch.mean(div)
		return div

	def div_loss(self, x, y, r1_gamma=10.0, cuda=False):
		x_ = x.requires_grad_(True)
		y_ = self.dis(x_)
		grad = autograd.grad(
			outputs=y_,
			inputs=x_,
			grad_outputs=torch.ones_like(y_),
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
			)[0]
		grad = grad * grad
		grad = grad.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
		loss = grad.mean()
		return loss

	def discriminator_latent_loss(self, real_pred, fake_pred):
		real_loss = F.softplus(-real_pred).mean()
		fake_loss = F.softplus(fake_pred).mean()
		return real_loss + fake_loss