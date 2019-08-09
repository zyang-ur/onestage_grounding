import torch
import numpy as np
import os

# from plot_util import plot_confusion_matrix
# from makemask import *

def _fast_hist(label_true, label_pred, n_class):
	mask = (label_true >= 0) & (label_true < n_class)
	hist = np.bincount(
		n_class * label_true[mask].astype(int) +
		label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
	return hist

def label_accuracy_score(label_trues, label_preds, n_class, bg_thre=200):
	"""Returns accuracy score evaluation result.
	  - overall accuracy
	  - mean accuracy
	  - mean IU
	  - fwavacc
	"""
	hist = np.zeros((n_class, n_class))
	for lt, lp in zip(label_trues, label_preds):
		# hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
		hist += _fast_hist(lt[lt<bg_thre].flatten(), lp[lt<bg_thre].flatten(), n_class)
	acc = np.diag(hist).sum() / hist.sum()
	acc_cls = np.diag(hist) / hist.sum(axis=1)
	acc_cls = np.nanmean(acc_cls)
	iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
	mean_iu = np.nanmean(iu)
	freq = hist.sum(axis=1) / hist.sum()
	fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
	return acc, acc_cls, mean_iu, fwavacc

def label_confusion_matrix(label_trues, label_preds, n_class, bg_thre=200):
	# eps=1e-20
	hist=np.zeros((n_class,n_class),dtype=float)
	""" (8,256,256), (256,256) """
	for lt,lp in zip(label_trues, label_preds):
		# hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
		hist += _fast_hist(lt[lt<bg_thre].flatten(), lp[lt<bg_thre].flatten(), n_class)
	iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
	# for i in range(n_class):
	# 	hist[i,:]=(hist[i,:]+eps)/sum(hist[i,:]+eps)
	return hist, iu

def body_region_confusion_matrix(label_trues, label_preds, n_class, boxes, counter):
	## pred: [bb,region_index,c,h,w] (pred score)
	## gt: [bb,region_index,h,w] (0-nclass score)
	label_trues = label_trues.data.cpu().numpy()
	label_preds = label_preds.data.cpu().numpy()
	hist=np.zeros((label_trues.shape[1],n_class,n_class),dtype=float)
	for body_i in range(label_trues.shape[1]):
		for bb in range(label_trues.shape[0]):
			if body_i != label_trues.shape[1]-1 and \
				torch.equal(boxes[bb,body_i,:], torch.Tensor([0.,0.,1.,1.])):
				counter+=1
				continue
			else:
				hist[body_i,:,:] += label_confusion_matrix(label_trues[bb,body_i,:,:], \
						np.argmax(label_preds[bb,body_i,:,:,:], axis=0), n_class)[0]
	return hist

def hist_based_accu_cal(hist):
	acc = np.diag(hist).sum() / hist.sum()
	acc_cls = np.diag(hist) / hist.sum(axis=1)
	acc_cls = np.nanmean(acc_cls)
	iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
	mean_iu = np.nanmean(iu)
	freq = hist.sum(axis=1) / hist.sum()
	fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
	return acc, acc_cls, mean_iu, fwavacc, iu

# if __name__ == '__main__':
# 	""" Evaluating from saved png segmentation maps 
# 		0.862723060822 0.608076070823 0.503493670787 0.76556929118
# 	"""
# 	import csv
# 	from PIL import Image
# 	import matplotlib as mpl
# 	mpl.use('Agg')
# 	from matplotlib import pyplot as plt
# 	eps=1e-20

# 	class AverageMeter(object):
# 		"""Computes and stores the average and current value"""
# 		def __init__(self):
# 			self.reset()

# 		def reset(self):
# 			self.val = 0
# 			self.avg = 0
# 			self.sum = 0
# 			self.count = 0

# 		def update(self, val, n=1):
# 			self.val = val
# 			self.sum += val * n
# 			self.count += n
# 			self.avg = self.sum / self.count
# 	def load_csv(csv_file):
# 		img_list, kpt_list, conf_list=[],[],[]
# 		with open(csv_file, 'rb') as f:
# 			reader = csv.reader(f)
# 			for row in reader:
# 				img_list.append(row[0])
# 				kpt_list.append([row[i] for i in range(1,len(row)) if i%3!=0])
# 				conf_list.append([row[i] for i in range(1,len(row)) if i%3==0])
# 		# print len(img_list),len(kpt_list[0]),len(conf_list[0])
# 		return img_list,kpt_list,conf_list

# 	n_class = 7
# 	superpixel_smooth = False
# 	# valfile = '../../ln_data/LIP/TrainVal_pose_annotations/lip_val_set.csv'
# 	# pred_folder = '../../../git_code/LIP_JPPNet/output/parsing/val/'
# 	# pred_folder = '../visulizations/refinenet_baseline/test_out/'
# 	pred_folder = '../visulizations/refinenet_splittask/test_out/'
# 	gt_folder = '../../ln_data/pascal_data/SegmentationPart/'
# 	img_path = '../../ln_data/pascal_data/JPEGImages/'

# 	file = '../../ln_data/pascal_data/val_id.txt'
# 	missjoints = '../../ln_data/pascal_data/no_joint_list.txt'
# 	img_list = [x.strip().split(' ')[0] for x in open(file)]
# 	miss_list = [x.strip().split(' ')[0] for x in open(missjoints)]

# 	conf_matrices = AverageMeter()
# 	for index in range(len(img_list)):
# 		img_name = img_list[index]
# 		if img_name in miss_list:
# 			continue
# 		if not os.path.isfile(pred_folder + img_name + '.png'):
# 			continue
# 		pred_file = pred_folder + img_name + '.png'
# 		pred = Image.open(pred_file)
# 		gt_file = gt_folder + img_name + '.png'
# 		gt = Image.open(gt_file)
# 		pred, gt = np.array(pred, dtype=np.int32), np.array(gt, dtype=np.int32)
# 		if superpixel_smooth:
# 			img_file = img_path+img_name+'.jpg'
# 			img = Image.open(img_file)
# 			pred = superpixel_expand(np.array(img),pred)
# 		confusion, _ = label_confusion_matrix(gt, pred, n_class)
# 		conf_matrices.update(confusion,1)
# 	acc, acc_cls, mean_iu, fwavacc, iu = hist_based_accu_cal(conf_matrices.avg)
# 	print(acc, acc_cls, mean_iu, fwavacc)
# 	print(iu)

# 	## SAVE CONFUSION MATRIX
# 	figure=plt.figure()
# 	class_name=['bg', 'head', 'torso', 'upper arm', 'lower arm', 'upper leg', 'lower leg']
# 	conf_matrices = conf_matrices.avg
# 	for i in range(n_class):
# 		conf_matrices[i,:]=(conf_matrices[i,:]+eps)/sum(conf_matrices[i,:]+eps)
# 	plot_confusion_matrix(conf_matrices, classes=class_name,
# 		rotation=0, include_text=True,
# 		title='Confusion matrix, without normalization')
# 	plt.show()
# 	plt.savefig('../saved_models/Baseline_refinenet_test.jpg')
# 	plt.close('all')
