import numpy as np
import pandas as pd




class Training_Dataset(object):
	"""docstring for Test"""
	def __init__(self,file):
		# self.arg = arg
		self.file = file
		self.counter = -1

	def read_data(self,X_index,Y_index,one_hot = False,n_classes = 0,sl = 0):
		data = pd.read_csv(self.file)

		self.X_i = data.iloc[:,sl].values

		self.X = data.iloc[:,X_index].values
		self.Y = data.iloc[:,Y_index].values

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


		# Y = (data.iloc[:,Y_index[:3]].values+data.iloc[:,Y_index[3:]].values)*0.5
		# Y = data.iloc[:,Y_index].values
		# data = data.drop(1,axis = 1)
	def std_data(self,X_index):
		X_std = np.copy(self.X)
		for i in xrange(len(X_index)) :
			mu = X[:,i].mean()
			std = X[:,i].std()
			X_std[:,i] = (X[:,i] - mu) / (std)
		self.X = X_std
		

	def reset_counter(self):	
		self.counter = -1
	def next_batch(self,n):
		self.counter += 1
		return self.X[self.counter*n:(self.counter+1)*n],self.Y[self.counter*n:(self.counter+1)*n], self.X_i




class TestDataset(object):
	"""docstring for Test"""
	def __init__(self,file):
		# self.arg = arg
		self.file = file
		self.counter = 0

	def read_data(self,X_index,sl = 0):
		data = pd.read_csv(self.file)

		self.X_i = data.iloc[:,0].values

		self.X = data.iloc[:,X_index].values

		self.SizE = len(self.X)
		# Y = (data.iloc[:,Y_index[:3]].values+data.iloc[:,Y_index[3:]].values)*0.5
		# Y = data.iloc[:,Y_index].values
		# data = data.drop(1,axis = 1)
	def std_data(self,X_index):
		X_std = np.copy(self.X)
		for i in xrange(len(X_index)) :
			mu = X[:,i].mean()
			std = X[:,i].std()
			X_std[:,i] = (X[:,i] - mu) / (std)
		self.X = X_std
		

	def reset_counter(self):	
		self.counter = 0
	def next_batch(self,n):
		k = self.counter
		self.counter += n
		return self.X[k:k+n], self.X_i