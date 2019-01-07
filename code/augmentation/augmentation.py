import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

def main():
	base_path = "/home/ml/ml_work01"
	image_path = "proc_images"
	output_image_path = os.path.join(base_path,"aug_images")
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

		index = 0	
		for image_name in os.listdir(os.path.join(image_level, level_name)):
			index += 1
			image = os.path.join(image_level, level_name, image_name)
			pre, ext = os.path.splitext(image_name)			
			im_data = cv2.imread(image)
			cp_image = os.path.join(output_sublevel_path,image_name)
			print cp_image
			cv2.imwrite(cp_image,im_data)

			# cv2.imshow('image', im_data)
			# cv2.waitKey(0)
			aug_seq = iaa.Sequential(
		            [
		                iaa.Add((-20, 20)),
		                iaa.ContrastNormalization((0.8, 1.6)),
		                iaa.AddToHueAndSaturation((-21, 21)),
		                iaa.SaltAndPepper(p=0.1),
		                iaa.Scale({"width":224, "height":"keep-aspect-ratio"}, 1),
		                iaa.CropAndPad(
			                percent=(-0.05, 0.1),
			                pad_mode=ia.ALL,
			                pad_cval=(0, 255)
		            	)
		            ],
		            random_order=True)

			for subindex in range(20):
				im_data_aug = aug_seq.augment_image(im_data)	
				cv2.imshow('augmantation', im_data_aug)
				new_image = pre + "_" + str(subindex) + ext
				new_image = os.path.join(output_sublevel_path,new_image)
				print new_image
				cv2.imwrite(new_image,im_data_aug)
				# cv2.waitKey(0)




if __name__ == "__main__":
	main()