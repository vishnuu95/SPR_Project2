from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
import numpy as np
import argparse
from joblib import dump, load
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='SVM train and test')
parser.add_argument('--phase', type=str, default="train", metavar='N',
                    help='test or train')
parser.add_argument('--save_model', type=bool, default=True, metavar='N',
                    help='save model or not')
# parser.add_argument('--epochs', type=int, default=14, metavar='N',
#                     help='number of epochs to train (default: 14)')
# parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                     help='learning rate (default: 1.0)')
# parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                     help='Learning rate step gamma (default: 0.7)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--dry-run', action='store_true', default=False,
#                     help='quickly check a single pass')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--save-model', action='store_true', default=False,
#                     help='For Saving the current Model')
args = parser.parse_args()

if(args.phase == "train"):
	mnist = fetch_openml('mnist_784', data_home="data/")
	print(mnist.data.shape)
	print(mnist.target.shape)

	np.random.seed(0)
	split = 0.8
	idx = np.random.shuffle(np.arange(mnist.data.shape[0]))
	data = np.squeeze(mnist.data[idx,:])
	# print(data.shape)
	labels = np.squeeze(mnist.target[idx])
	train_data = data[:int(split*data.shape[0]), :]
	test_data = data[int(split*data.shape[0]):, :]
	train_labels = labels[:int(split*data.shape[0])]
	test_labels = labels[int(split*data.shape[0]):]
	# train_data = np.squeeze(train_data, axis=0)
	classifier = svm.SVC(decision_function_shape='ovo')
	print(train_data.shape, train_labels.shape)
	classifier.fit(train_data, train_labels)
	if(args.save_model == True):
		dump(classifier, 'models/svm_trained.joblib')
	predicts = classifier.predict(test_data)	
	print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test_labels, predicts)))
	disp = metrics.plot_confusion_matrix(classifier, test_data, test_labels)
	disp.figure_.suptitle("Confusion Matrix")
	print("Confusion matrix:\n%s" % disp.confusion_matrix)
