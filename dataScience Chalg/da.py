import numpy as np
import pandas as pd
from math import log
file = 'data/movies.csv'
data = pd.read_csv(file)

X = data.iloc[:,2].values
y = data.iloc[:,0].values
# print(X[:2])
X = map(lambda x:x.split('|'),X)
# print(X[:2])

movies_type = []
movies_type.append([j for i in X for j in i ])
movies_type = list(set(*movies_type))
n_movies_type =  len(movies_type)

# print(movies_type)
# print(movies_type[0])
movies_dict = {}
for i in xrange(n_movies_type):
	movies_dict[movies_type[i]] = i




def one_hot(X,n):
	one_h = np.zeros(n)
	for i in xrange(len(X)):
		try:
			one_h[movies_dict[X[i]]] = 1
		except Exception as e:
			pass
	return one_h

movies_data = {}
for i in xrange(len(X)):
	one_hot_vector = one_hot(X[i],n_movies_type)
	movies_data[y[i]] = one_hot_vector

# movies_data = np.array(movies_data)
# movies_data.in_csv('data/movies_oh.csv')
# movies_data[2]+=movies_data[2]
# print(n_movies_type,movies_data[2]*4)

###############################################################################
file = 'data/training.csv'
data = pd.read_csv(file)

userId = data.iloc[:,3].values
movies_id = data.iloc[:,0].values
rating = data.iloc[:,1].values

user_dict_data = {}
user_dict_n = {}
for itr in xrange(len(userId)):
	try:
		user_dict_data[userId[itr]] += movies_data[int(movies_id[itr])]*rating[itr]
		user_dict_n[userId[itr]] += 1
	except Exception as e:
		user_dict_data[userId[itr]] = np.array(movies_data[int(movies_id[itr])]*rating[itr])
		user_dict_n[userId[itr]] = 1

# print(user_dict_data[10917565],user_dict_n[10917565])


##################################################################################

def Round(n):
	# print n
	frac = n-int(n)
	if(frac>0.25 and frac<0.75):
		f = 0.5
	elif frac<=0.25:
		f = 0.0
	elif frac>=0.75:
		f = 1.0
	return int(n)+f

def log_likely(a,b):

	# y = (a-a.mean())/a.std()
	y = np.array(map(float,a))
	y = y/max(y)
	y = np.where(y<=0,1e-10,y)
	sc = 0
	for i in range(len(b)):
		sc += b[i]*log(y[i]) + (1-b[i])*log(1-y[i]+1e-10)
	return sc

def top_5(userId):
	score = []
	ct = 0
	for key in movies_data.keys():
		score.append([key,log_likely(user_dict_data[userId],movies_data[key])])
	score = sorted(score,key = lambda x:x[1],reverse = True)
	score = score[:5]
	movie_recom_rat = [[score[i][0],sum(movies_data[score[i][0]]*user_dict_data[userId])/sum(user_dict_data[userId])] for i in xrange(len(score))]
	# movie_recom_rat = [[score[i][0],sum(movies_data[score[i][0]]*user_dict_data[userId])/sum(movies_data[score[i][0]])] for i in xrange(len(score))]
	
	rat_list = [i/2.0 for i in xrange(11)]
	movie_recom_rat = sorted(movie_recom_rat,key = lambda x:x[1],reverse = True)
	x0 = movie_recom_rat[0][1]
	y0 = 5.0
	

	movie_rat = []
	# print movie_recom_rat
	for i in range(5):
		if movie_recom_rat[0][1] - movie_recom_rat[4][1]:
			m = 3.0/(movie_recom_rat[0][1] - movie_recom_rat[4][1])
			rat = m*(movie_recom_rat[i][1]-x0)+y0
		else:
			rat = 5.0
		# print rat
		movie_rat.append([score[i][0],Round(rat)])

	return movie_rat


file = 'data/test.csv'
data = pd.read_csv(file)
userId_test = data.iloc[:,0].values



tp5 = top_5(userId_test[5])
print tp5
# a =[]
# for i in range(5):
# 	a.append(movies_data[tp5[i][0]])

# a.append(user_dict_data[userId_test[0]])
# for i in range(len(a[0])):
# 	print(a[0][i],a[1][i],a[2][i],a[3][i],a[4][i],a[5][i])

# print(user_dict_data[userId_test[5]],user_dict_n[userId_test[5]])  

f = open('test_o.csv','w')
f.write("userId,movieId,rating\n")
llt = len(userId_test)
# for i in range(10):
for i in range(len(userId_test)):
	try:
		tp5 = top_5(userId_test[i])
	except:
		pass
	print(1.0*i/llt," % done")
	for j in range(5):
		rrt = str(tp5[j][1])
		try:
			if int(rrt.split('.')[1]) == 0:
				rrt = rrt.split('.')[0]
		except:
			pass
		# print(str(userId_test[i])+','+str(tp5[j][0])+','+str(rrt))
		f.write(str(userId_test[i])+','+str(tp5[j][0])+','+str(rrt)+'\n')

# f.close()
