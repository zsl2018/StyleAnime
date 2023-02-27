# Parsing-Conditioned Anime Translation: A New Dataset and Method(ACM TOG)
> Anime is an abstract art form that is substantially different from the human portrait, leading to a challenging misaligned image translation problem that is beyond the capability of existing methods. This can be boiled down to a highly ambiguous unconstrained translation between two domains. To this end, we design a new anime translation framework by deriving the prior knowledge of a pre-trained StyleGAN model. We introduce disentangled encoders to separately embed structure and appearance information into the same latent code, governed by four tailored losses. Moreover, we develop a FaceBank aggregation method that leverages the generated data of the StyleGAN, anchoring the prediction to, produce in-domain animes. To empower our model and promote the research of anime translation, we propose the first anime portrait parsing dataset, Danbooru-Parsing, containing 4,921 densely labeled images across 17 classes. This dataset connects the face semantics with appearances, enabling our new constrained translation setting. We further show the editability of our results, and extend our method to manga images, by generating the first manga parsing pseudo data. Extensive experiments demonstrate the values of our new dataset and method, resulting in the first feasible solution on anime translation.
![image](https://github.com/zsl2018/StyleAnime/blob/master/Images/overview.png)

# Description
This is the official implementation of our paper "Parsing-Conditioned Anime Translation: A New Dataset and Method"(ACM TOG).

# Pretrained Models
Please download the pre-trained models from the following links.  
｜       Path     |    Description      |  
| [StyleAnime](https://drive.google.com/file/d/1zresf1KfX1keJT2dv0jZesDppWLQXYNa/view?usp=share_link)   | Our pretrained styleAnime model (portrait2anime)|  

The pretrained models should be saved to the directory pretrained_models.
d 
