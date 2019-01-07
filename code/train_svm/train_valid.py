import os
import cv2
import numpy as np

def main():
	features = np.load("cat.npz")
	print features["f1"].shape

if __name__ == "__main__":
	main()