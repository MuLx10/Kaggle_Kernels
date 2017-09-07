import numpy as np
import pandas as pd




class TrainingDataset(object):
	"""docstring for Test"""
	def __init__(self,file):
		# self.arg = arg
		self.file = file
		self.counter = -1

	def read_data(self,X_index,Y_index,one_hot = False,n_classes = 0,sl = 0):
		data = pd.read_csv(self.file)

		# self.X_i = data.iloc[:,sl].values

		self.X = data.iloc[:,X_index].values
		self.Y = data.iloc[:,Y_index].values

		# self.X[:][]
		self.X[:,0] = 6/self.X[:,0]
		self.X[:,1]=np.where(self.X[:,1]=='female',1,0)
		self.X[:,2]=self.X[:,2]+self.X[:,3]
		self.X[:,3]=self.X[:,4]

		# self.X.drop(self.X.index[5])
		self.X = np.delete(self.X,np.s_[4],1)

		# nan_rx = ~np.isnan(np.array(self.X)).any(axis=1)
		# nan_ry = ~np.isnan(np.array(self.Y)).any(axis=1)
		# nan_r = nan_ry+nan_rx

		# self.X = self.X[nan_r]
		# self.Y = self.Y[nan_r]

		self.SIZE = len(self.X)

		if one_hot:
			Y = []
			for i in xrange(len(self.Y)):
				temp = []
				[temp.append(0) for _ in range(n_classes)]
				# print(self.Y[i])
				temp[self.Y[i]] = 1
				Y.append(temp)
			self.Y = Y
		# else:
		# 	Y = []
		# 	for i in xrange(len(self.Y)):
		# 		Y.append([self.Y[i]])
		# 	self.Y = Y


		# Y = (data.iloc[:,Y_index[:3]].values+data.iloc[:,Y_index[3:]].values)*0.5
		# Y = data.iloc[:,Y_index].values
		# data = data.drop(1,axis = 1)
	def std_data(self,X_index):
		X_std = np.copy(self.X)
		for i in xrange(len(X_index)-1) :
			mu = self.X[:,i].mean()
			std = self.X[:,i].std()
			X_std[:,i] = (self.X[:,i] - mu) / (std)
		self.X = X_std
		

	def reset_counter(self):	
		self.counter = -1
	def next_batch(self,n):
		self.counter += 1
		return self.X[self.counter*n:(self.counter+1)*n],self.Y[self.counter*n:(self.counter+1)*n]




class TestDataset(object):
	"""docstring for Test"""
	def __init__(self,file):
		# self.arg = arg
		self.file = file
		self.counter = -1

	def read_data(self,X_index,sl=0):
		data = pd.read_csv(self.file)

		self.X_i = data.iloc[:,sl].values

		self.X = data.iloc[:,X_index].values
		# self.Y = data.iloc[:,Y_index].values

		# self.X[:][]
		self.X[:,0] = 6/self.X[:,0]
		self.X[:,1]=np.where(self.X[:,1]=='female',1,0)
		self.X[:,2]=self.X[:,2]+self.X[:,3]
		self.X[:,3]=self.X[:,4]

		# self.X.drop(self.X.index[5])
		self.X = np.delete(self.X,np.s_[4],1)

		self.SIZE = len(self.X)

		# if one_hot:
		# 	Y = []
		# 	for i in xrange(len(self.Y)):
		# 		temp = []
		# 		[temp.append(0) for _ in range(n_classes)]
		# 		# print(self.Y[i])
		# 		temp[self.Y[i]] = 1
		# 		Y.append(temp)
		# 	self.Y = Y


		# Y = (data.iloc[:,Y_index[:3]].values+data.iloc[:,Y_index[3:]].values)*0.5
		# Y = data.iloc[:,Y_index].values
		# data = data.drop(1,axis = 1)
	def std_data(self,X_index):
		X_std = np.copy(self.X)
		for i in xrange(len(X_index)-1) :
			mu = self.X[:,i].mean()
			std = self.X[:,i].std()
			X_std[:,i] = (self.X[:,i] - mu) / (std)
		self.X = X_std
		

	def reset_counter(self):	
		self.counter = -1
	def next_batch(self,n):
		self.counter += 1
		return self.X[self.counter*n:(self.counter+1)*n],self.X_i[self.counter*n:(self.counter+1)*n]