import os
import cv2

def main():
	base_path = "/home/ml/ml_work01"
	image_path = "selected_images"
	output_image_path = os.path.join(base_path,"proc_images")
	if not (os.path.exists(output_image_path)):
		os.mkdir(output_image_path,0777)

	image = []

	image_level = os.path.join(base_path,image_path)
	image_sublevel = os.listdir(image_level)
	
	for level_name in image_sublevel:
		output_sublevel_path = os.path.join(output_image_path,level_name)
		if not (os.path.exists(output_sublevel_path)):
			os.mkdir(output_sublevel_path,0777)
		index = 0	
		for image_name in os.listdir(os.path.join(image_level, level_name)):
			index += 1
			image = os.path.join(image_level, level_name, image_name)			
			im_data = cv2.imread(image)	
			if im_data.shape[0] < im_data.shape[1]:
			 	im_data = cv2.transpose(im_data)
				im_data = cv2.flip(im_data, flipCode=1)
			im_data = cv2.resize(im_data, (224,224))
#			im_data = im_data / 255.
			name = level_name + "_"+ str(index) + ".jpg"
			name = os.path.join(output_sublevel_path,name)
			print name
			# cv2.imshow('image',im_data)
			cv2.imwrite(name,im_data)
			# cv2.waitKey(0)			


if __name__ == "__main__":
	main()