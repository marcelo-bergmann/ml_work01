import os
import cv2
import keras.backend as K
from utils import load_model
import numpy as np

def load_convnet():
	model = load_model()
	model.summary()
	return model

def main():
	net = load_convnet()
	get_features = K.function([net.layers[0].input, K.learning_phase()], [net.get_layer("flatten_2").output])

	base_path = "/home/ml/ml_work01"
	image_path = "aug_images"
	output_image_path = os.path.join(base_path,"feature")
	if not (os.path.exists(output_image_path)):
		os.mkdir(output_image_path,0777)

	image_level = os.path.join(base_path,image_path)
	image_sublevel = os.listdir(image_level)
	# print image_level
	# print image_sublevel

	for level_name in image_sublevel:
		output_sublevel_path = os.path.join(output_image_path,level_name)
		if not (os.path.exists(output_sublevel_path)):
			os.mkdir(output_sublevel_path,0777)

		for image_name in os.listdir(os.path.join(image_level, level_name)):
			image = os.path.join(image_level, level_name, image_name)
			print image
			pre, ext = os.path.splitext(image_name)	

			im_data = cv2.imread(image)
			features = get_features([im_data[np.newaxis,...],0])[0]
			print features.shape
			# cv2.imshow("frameB",im_data)

			feature_file = pre + ".npz"
			feature_file = os.path.join(output_sublevel_path,feature_file)
			print feature_file
			np.savez_compressed(feature_file,**{"f1":features})
			# cv2.waitKey(0)


if __name__ == "__main__":
	main()
