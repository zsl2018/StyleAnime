from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'anime_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['anime_train_segmentation'],
		'train_target_root': dataset_paths['anime_train'],
		'test_source_root': dataset_paths['anime_test_segmentation'],
		'test_target_root': dataset_paths['anime_test'],
		'face_train_source_root': dataset_paths['face_train_segmentation'],
		'face_train_target_root': dataset_paths['face_train'],
		'face_test_source_root': dataset_paths['face_test_segmentation'],
		'face_test_target_root': dataset_paths['face_test'],
	}
}
