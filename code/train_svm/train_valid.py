import os
import cv2
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics

def main():

	base_path = "/home/ml/ml_work01"
	feature_path = "feature"

	feature_level = os.path.join(base_path,feature_path)
	feature_sublevel = os.listdir(feature_level)
	print feature_level
	print feature_sublevel

	all_features = []
	all_classes = []

	for class_name in feature_sublevel:
		for feature_name in os.listdir(os.path.join(feature_level, class_name)):
			current_feature = os.path.join(feature_level, class_name, feature_name)
			# print current_feature
			features = np.load(current_feature)
			# print features["f1"].shape
			all_features.append(features["f1"][0])
			# print all_features[-1].shape
			all_classes.append(class_name)
			features.close()

	# print all_classes

		# X_train, X_test, y_train, y_test = train_test_split(all_features, all_classes, test_size=0.2,random_state=193)
		# clf = svm.SVC(kernel='linear', C = 1.0)
	print "===== Method SVC ====="
	for index in range(3):
		X_train, X_test, y_train, y_test = train_test_split(all_features, all_classes, test_size=0.2,random_state=index)
		clf = svm.SVC(kernel='linear', C = 1.0, verbose=1)

		print "Training SVC" + str(index)
		clf.fit(X_train, y_train)
		print "Prediction: "
		y_pred = clf.predict(X_test)
		acc = metrics.accuracy_score(y_test, y_pred)
		print "Accuracy: " + str(acc)
		prec = metrics.precision_score(y_test, y_pred, average='weighted')
		print "Precision: " + str(prec)
		recall = metrics.recall_score(y_test, y_pred, average='weighted') 
		print "Recall: " + str(recall)
		print "----------------------------"

		# print('w = ',clf.coef_)
		# print('b = ',clf.intercept_)
		# print('Indices of support vectors = ', clf.support_)
		# print('Support vectors = ', clf.support_vectors_)
		# print('Number of support vectors for each class = ', clf.n_support_)
		# print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))

	print "===== Method LinearSVC ====="
	for index in range(3):
		X_train, X_test, y_train, y_test = train_test_split(all_features, all_classes, test_size=0.2,random_state=index)
		clf = svm.LinearSVC(random_state=0, tol=1e-5, verbose=1)

		print "Training SVC" + str(index)
		clf.fit(X_train, y_train)
		print "Prediction: "
		y_pred = clf.predict(X_test)
		acc = metrics.accuracy_score(y_test, y_pred)
		print "Accuracy: " + str(acc)
		prec = metrics.precision_score(y_test, y_pred, average='weighted')
		print "Precision: " + str(prec)
		recall = metrics.recall_score(y_test, y_pred, average='weighted') 
		print "Recall: " + str(recall)
		print "----------------------------"

		# print('w = ',clf.coef_)
		# print('b = ',clf.intercept_)
		# print('Indices of support vectors = ', clf.support_)
		# print('Support vectors = ', clf.support_vectors_)
		# print('Number of support vectors for each class = ', clf.n_support_)
		# print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))


if __name__ == "__main__":
	main()