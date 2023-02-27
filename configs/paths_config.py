dataset_paths = {
	'anime_train_segmentation': '/mnt/lab/lzs/psp_anime_4921/anime/anime_seg_train',
	'anime_test_segmentation': '/mnt/lab/lzs/psp_anime_4921/anime/anime_seg_test_68',
	'anime_train': '/mnt/lab/lzs/psp_anime_4921/anime/anime_face_train',
	'anime_test': '/mnt/lab/lzs/psp_anime_4921/anime/anime_face_test_68',
    
	'face_train_segmentation': '/mnt/lab/lzs/psp_anime_4921/celeba/celeba_seg_train',
	'face_test_segmentation': '/mnt/lab/lzs/psp_anime_4921/celeba/celeba_seg_test_68',
	'face_train': '/mnt/lab/lzs/psp_anime_4921/celeba/celeba_face_train',
	'face_test': '/mnt/lab/lzs/psp_anime_4921/celeba/celeba_face_test_68',
}

model_paths = {
	'anime_ffhq': 'pretrained_models/stylegan2_anime_pretrained.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat'
}
