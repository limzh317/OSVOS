import SimpleITK as sitk
import numpy as np
import cv2
import os

# import time from PIL
# import Image
a = []
b = []


def file_name(file_dir):
	for root, dirs, files in os.walk(file_dir):
		a.append(root)
		b.append(dirs)


count = 1
path = "/media/zeven/ba632b01-5087-4730-8be4-c68038e7e90f/YF/cd1/N11947829_0001813007/"
file_name(path)
j = 13
p = a[j]
filename = os.listdir(p)
count = 1
os.mkdir("/media/zeven/ba632b01-5087-4730-8be4-c68038e7e90f/dcmtojpg/cd1/" + b[0][j - 1])
for i in filename:
	document = os.path.join(p, i)
	outputpath = "/media/zeven/ba632b01-5087-4730-8be4-c68038e7e90f/dcmtojpg/cd1/" + b[0][j - 1]
	countname = i[0:-4]
	countfullname = countname + '.jpg'

	output_jpg_path = os.path.join(outputpath, countfullname)


	def convert_from_dicom_to_jpg(img, low_window, high_window, save_path):
		lungwin = np.array([low_window * 1., high_window * 1.])
		newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
		newimg = (newimg * 255).astype('uint8')
		cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


	if __name__ == '__main__':
		ds_array = sitk.ReadImage(document)
		img_array = sitk.GetArrayFromImage(ds_array)

		shape = img_array.shape  # name.shape
		if len(shape) == 3:
			img_array = np.reshape(img_array, (shape[1], shape[2]))
		if len(shape) == 4:
			img_array = np.reshape(img_array, (shape[1], shape[2], shape[3]))
		high = np.max(img_array)
		low = np.min(img_array)
		convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)
		print(j, len(a), count)
	count += 1
