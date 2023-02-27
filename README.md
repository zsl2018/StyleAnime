# Parsing-Conditioned Anime Translation: A New Dataset and Method(ACM TOG)
> Anime is an abstract art form that is substantially different from the human portrait, leading to a challenging misaligned image translation problem that is beyond the capability of existing methods. This can be boiled down to a highly ambiguous unconstrained translation between two domains. To this end, we design a new anime translation framework by deriving the prior knowledge of a pre-trained StyleGAN model. We introduce disentangled encoders to separately embed structure and appearance information into the same latent code, governed by four tailored losses. Moreover, we develop a FaceBank aggregation method that leverages the generated data of the StyleGAN, anchoring the prediction to, produce in-domain animes. To empower our model and promote the research of anime translation, we propose the first anime portrait parsing dataset, Danbooru-Parsing, containing 4,921 densely labeled images across 17 classes. This dataset connects the face semantics with appearances, enabling our new constrained translation setting. We further show the editability of our results, and extend our method to manga images, by generating the first manga parsing pseudo data. Extensive experiments demonstrate the values of our new dataset and method, resulting in the first feasible solution on anime translation.
![image](https://github.com/zsl2018/StyleAnime/blob/master/Images/overview.png)

# Description
This is the official implementation of our paper "Parsing-Conditioned Anime Translation: A New Dataset and Method"(ACM TOG).

# Pretrained Models
Please download the pre-trained models from the following links.  
      Path     |    Description      |  
|----------------|--------------------|
| [StyleAnime](https://drive.google.com/file/d/1zresf1KfX1keJT2dv0jZesDppWLQXYNa/view?usp=share_link)   | Our pretrained styleAnime model (portrait2anime)|  | [IR-SE50 Model](https://drive.google.com/file/d/1b7d9xyvUm1y2xxMyX_LTy9UgamVf563F/view?usp=sharing) |Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during pSp training.| 
| [MTCNN](https://drive.google.com/file/d/1w46525L0FvoCzcVZpuiXVfMzmyOO_3A2/view?usp=sharing) | Weights for [MTCNN](https://github.com/TreB1eN/InsightFace_Pytorch) model taken from TreB1eN for use in ID similarity metric computation. (Unpack the tar.gz to extract the 3 model weights.)|
|[CurricularFace Backbone](https://drive.google.com/file/d/1y5Y7ZVZNd9UAqVkXnGwl9NK10SUUjsr4/view?usp=share_link) | Pretrained CurricularFace model taken from [HuangYG123](https://github.com/HuangYG123/CurricularFace) for use in ID similarity metric computation.| 
| [Anime StyleGAN2 Model] | Pretrained model of StyleGAN2 on our anime dataset | 
| [Average Latent ] | Average latent of Stylegan2 pretrained model on anime | 
| [Bank list] | FaceBank Aggregation, stylegan2 anime latent list|


The pretrained models should be saved to the directory pretrained_models.

# Preparing Data
Please go to configs/paths_config.py and define:
```
dataset_paths = {
	'anime_train_segmentation': 'path/anime/anime_seg_train',
	'anime_test_segmentation': 'path/anime/anime_seg_test_68',
	'anime_train': 'path/anime/anime_face_train',
	'anime_test': 'path/anime_face_test_68',
    
	'face_train_segmentation': 'path/celeba/celeba_seg_train',
	'face_test_segmentation': 'path/celeba/celeba_seg_test_68',
	'face_train': 'path/celeba/celeba_face_train',
	'face_test': 'path/celeba/celeba_face_test_68',
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

