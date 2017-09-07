import numpy as np
import pandas as pd

file = 'data/train.csv'
data = pd.read_csv(file)

userId = data.iloc[:,3].values
movies_id = data.iloc[:,0].values
rating = data.iloc[:,1].values

user_dict = {}

for itr in xrange(len(userId)):
	try:
		user_dict[userId[itr]] += movies_data[int(movies_id[itr])]*rating[itr]
	except Exception as e:
		user_dict[userId[itr]] = np.array(movies_data[int(movies_id[itr])]*rating[itr])
# print(X[:2])

# print(X[:2])

movies_type = []
movies_type.append([j for i in X for j in i ])
movies_type = list(set(*movies_type))
n_movies_type =  len(movies_type)

print(movies_type)
print(movies_type[0])
movies_dict = {}
for i in xrange(n_movies_type):
	movies_dict[movies_type[i]] = i




def one_hot(X,n):
	one_h = [0]*n
	for i in xrange(len(X)):
		try:
			one_h[movies_dict[X[i]]] = 1
		except Exception as e:
			pass
	return one_h

X_data = []
for i in xrange(len(X)):
	one_hot_vector = one_hot(X[i],n_movies_type)
	X_data.append([y[i],one_hot_vector])

# X_data = np.array(X_data)
# X_data.in_csv('data/movies_oh.csv')
print(n_movies_type,X_data[:2]+X_data[:2])

