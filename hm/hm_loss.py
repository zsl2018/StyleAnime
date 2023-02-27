import torch
from hm.histogram_matching import histogram_matching
import torch
from torch import nn
from torch.autograd import Variable
class HMLoss(nn.Module):
	def __init__(self, lambda_his_face=0.1, lambda_his_hair=0.1, lambda_his_eye=0.1):
		super(HMLoss, self).__init__()
		self.lambda_his_face = lambda_his_face
		self.lambda_his_hair = lambda_his_hair
		self.lambda_his_eye = lambda_his_eye
		self.criterionL1 = torch.nn.L1Loss()

	def to_var(self, x, requires_grad=True):
		if torch.cuda.is_available():
			x = x.cuda()
		if not requires_grad:
			return Variable(x, requires_grad=requires_grad)
		else:
			return Variable(x)
	def de_norm(self, x):
		out = (x + 1) / 2
		return out.clamp(0, 1)

	def mask_preprocess(self, mask_A, mask_B):
		index_tmp = mask_A.nonzero()
		x_A_index = index_tmp[:, 2]
		y_A_index = index_tmp[:, 3]
		index_tmp = mask_B.nonzero()
		x_B_index = index_tmp[:, 2]
		y_B_index = index_tmp[:, 3]
		mask_A = self.to_var(mask_A, requires_grad=False)
		mask_B = self.to_var(mask_B, requires_grad=False)
		index = [x_A_index, y_A_index, x_B_index, y_B_index]
		index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
		return mask_A, mask_B, index, index_2

	def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
		input_data = (self.de_norm(input_data) * 255).squeeze()
		target_data = (self.de_norm(target_data) * 255).squeeze()
		#print("mask_src", mask_src.shape)
		mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
		mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
		input_masked = input_data * mask_src
		target_masked = target_data * mask_tar
		# dstImg = (input_masked.data).cpu().clone()
		# refImg = (target_masked.data).cpu().clone()
		input_match = histogram_matching(input_masked, target_masked, index)
		input_match = self.to_var(input_match, requires_grad=False)
		loss = self.criterionL1(input_masked, input_match)
		return loss

	def rebound_box(self, mask_A, mask_B, mask_A_face):
		index_tmp = mask_A.nonzero()
		x_A_index = index_tmp[:, 2]
		y_A_index = index_tmp[:, 3]
		index_tmp = mask_B.nonzero()
		x_B_index = index_tmp[:, 2]
		y_B_index = index_tmp[:, 3]
		mask_A_temp = mask_A.copy_(mask_A)
		mask_B_temp = mask_B.copy_(mask_B)
		mask_A_temp[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
							mask_A_face[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
		mask_B_temp[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
							mask_A_face[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
		mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
		mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
		return mask_A_temp, mask_B_temp

	def forward(self, fake_A, ref_B, mask_A, mask_B):
		# Convert tensor to variable
		# ['hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
		#		'nose', 'mouth', 'u_face', 'l_face', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
		if True:
			if True:
				mask_A_face = (mask_A==1).float() + (mask_A==10).float() + (mask_A==14).float() + (mask_A==7).float() + (mask_A==8).float() + (mask_A==11).float()
				mask_B_face = (mask_B==1).float() + (mask_B==10).float() + (mask_B==14).float() + (mask_B==7).float() + (mask_B==8).float() + (mask_A==11).float()
				mask_A_face, mask_B_face, index_A_face, index_B_face = self.mask_preprocess(mask_A_face, mask_B_face)
			if True:
				mask_A_hair = (mask_A==17).float()
				mask_B_hair = (mask_B==17).float()
				mask_A_hair, mask_B_hair, index_A_hair, index_B_hair =self. mask_preprocess(mask_A_hair, mask_B_hair)
			if True:
				mask_A_eye_left = (mask_A==4).float() + (mask_A==2).float()
				mask_A_eye_right = (mask_A==5).float() + (mask_A==3).float()
				mask_B_eye_left = (mask_B==4).float() + (mask_B==2).float()
				mask_B_eye_right = (mask_B==5).float() + (mask_B==3).float()

				mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
					self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
				mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
					self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)

		g_A_loss_his = 0

		if True:
			if True:
				g_A_face_loss_his = self.criterionHis(fake_A, ref_B, mask_A_face, mask_B_face, index_A_face) * self.lambda_his_face
				g_A_loss_his += g_A_face_loss_his
			if True:
				g_A_hair_loss_his = self.criterionHis(fake_A, ref_B, mask_A_hair, mask_B_hair, index_A_hair) * self.lambda_his_hair
				g_A_loss_his += g_A_hair_loss_his
			if True:
				g_A_eye_left_loss_his = self.criterionHis(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye
				g_A_eye_right_loss_his = self.criterionHis(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye
				g_A_loss_his += g_A_eye_left_loss_his + g_A_eye_right_loss_his

		return g_A_loss_his
