import sklearn as sk
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
#from sklearn.cross_validation import cross_val_scor
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

"""
The task is pretty simple here. Apply your machine learning methods and analyze the results. 
You must produce the final output as "training_labels.txt" for the input "training_data.txt" and save the output as "testing_labels.txt"

Thank you for your effort. 

Deadline: 30th December,2017.

Submission Procedure: 1 week before deadline, you will have A goodgle Drive access to upload your assignment.

Do not forget to write  your roll number somewhere I find easily.

For any further question, send an email to; bishnukuet@gmail.com

Thank you.
"""



#TODO: Write your code here.



def cross_val(c,ltr,lts,l,l2):
	p = SelectKBest(f_classif, k=c)
	ltrn = p.fit_transform(ltr, l)
	ltsn = p.transform(lts)

	print "After feature selection......."

	print "\n\n\n"
	gnb = GaussianNB()
	y_pred = gnb.fit(ltrn, l).predict(ltsn)

	print "Naive Bayes: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(gnb, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	gx=scor.mean()

	# import matplotlib.pyplot as plt

	# Plot the data


	s = svm.SVC()
	y_pred = s.fit(ltrn, l).predict(ltsn)

	print "SVC: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(s, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	sx=scor.mean()

	sl = svm.LinearSVC()
	y_pred = sl.fit(ltrn, l).predict(ltsn)

	print "Liner SVC: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(sl, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()

	ls=scor.mean()

	ml = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=3000)
	y_pred = ml.fit(ltrn, l).predict(ltsn)

	print "Neural Net, Solver=adam: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(ml, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	nx1=scor.mean()
	ml = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=3000)
	y_pred = ml.fit(ltrn, l).predict(ltsn)

	print "Neural Net, Solver=lbfgs: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(ml, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	nx2=scor.mean()
	ml = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=3000)
	y_pred = ml.fit(ltrn, l).predict(ltsn)

	print "Neural Net, Solver=sgd: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(ml, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	nx3=scor.mean()
	sl = svm.LinearSVC()
	y_pred = sl.fit(ltrn, l).predict(ltsn)

	print "Liner SVC: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(sl, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()

	s = svm.SVC()
	y_pred = s.fit(ltrn, l).predict(ltsn)

	print "SVC: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(s, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()

	r = RandomForestClassifier(max_depth=4, random_state=0)
	y_pred = r.fit(ltrn, l).predict(ltsn)

	print "Random Forest: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(r, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	rx=scor.mean()
	n = KNeighborsClassifier(n_neighbors=1)
	y_pred = n.fit(ltrn, l).predict(ltsn)

	print "Nearest Neighbor: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(n, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	nx=scor.mean()
	print len(ltrn[0]), len(ltsn[0])

	return gx,sx,ls,nx1,nx2,nx3,rx,nx

def cross_val2(c,ltr,lts,l,l2):
	p = VarianceThreshold(threshold=c)
	ltrn = p.fit_transform(ltr, l)
	ltsn = p.transform(lts)

	print "After feature selection2......."

	print "\n\n\n"
	gnb = GaussianNB()
	y_pred = gnb.fit(ltrn, l).predict(ltsn)

	print "Naive Bayes: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(gnb, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	gx=scor.mean()

	# import matplotlib.pyplot as plt

	# Plot the data


	s = svm.SVC()
	y_pred = s.fit(ltrn, l).predict(ltsn)

	print "SVC: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(s, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	sx=scor.mean()

	sl = svm.LinearSVC()
	y_pred = sl.fit(ltrn, l).predict(ltsn)

	print "Liner SVC: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(sl, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()

	ls=scor.mean()

	ml = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=3000)
	y_pred = ml.fit(ltrn, l).predict(ltsn)

	print "Neural Net, Solver=adam: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(ml, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	nx1=scor.mean()
	ml = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=3000)
	y_pred = ml.fit(ltrn, l).predict(ltsn)

	print "Neural Net, Solver=lbfgs: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(ml, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	nx2=scor.mean()
	ml = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=3000)
	y_pred = ml.fit(ltrn, l).predict(ltsn)

	print "Neural Net, Solver=sgd: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(ml, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	nx3=scor.mean()
	sl = svm.LinearSVC()
	y_pred = sl.fit(ltrn, l).predict(ltsn)

	print "Liner SVC: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(sl, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()

	s = svm.SVC()
	y_pred = s.fit(ltrn, l).predict(ltsn)

	print "SVC: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(s, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()

	r = RandomForestClassifier(max_depth=4, random_state=0)
	y_pred = r.fit(ltrn, l).predict(ltsn)

	print "Random Forest: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(r, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	rx=scor.mean()
	n = KNeighborsClassifier(n_neighbors=1)
	y_pred = n.fit(ltrn, l).predict(ltsn)

	print "Nearest Neighbor: ", accuracy_score(l2, y_pred)
	print confusion_matrix(l2, y_pred, )

	scor = cross_validation.cross_val_score(n, ltrn, l, cv=5, scoring="accuracy")
	print "Cross - validated scores:", scor, " ", scor.mean()
	nx=scor.mean()
	print len(ltrn[0]), len(ltsn[0])
	aa,bb=ltrn.shape
	return gx,sx,ls,nx1,nx2,nx3,rx,nx,bb

if __name__ == '__main__':



	with open('training_data.txt') as f:
		d=f.readlines()

	with open('testing_data.txt') as f:
		d2=f.readlines()

	with open('training_labels.txt') as f:
		l=f.readlines()

	with open('testing_labels.txt') as f:
		l2=f.readlines()



	ltr=[]
	for p in d:
		j=p.split()
		j=np.array(j)
		j=j.astype(float)
		j=j.tolist()
		ltr.append(j)

	print ltr

	lts=[]
	for p in d2:
		j=p.split()
		j=np.array(j)
		j=j.astype(float)
		j=j.tolist()
		lts.append(j)

	print lts

	feature_name=['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter( %)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP',
				  'MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA',
				  'spread1','spread2','D2','PPE']




	p=VarianceThreshold(threshold=0.001)
	ltrnn =p.fit_transform(ltr, l)
	ltsnn= p.transform(lts)
	print "Feature Scores: ",ltrnn.shape



	map={}
	for i in range(len(feature_name)):
		#print feature_name[i]," = ",p.scores_[i]
		map[feature_name[i]]=p.variances_[i]

	aa,bbb=ltrnn.shape
	kk=[]
	for key, value in sorted(map.iteritems(), key=lambda (k, v): (v, k)):
		print key," ",value
		kk.append(key)
	best=[]
	i=1
	for key in reversed(kk):
		if i==bbb+1:
			break
		best.append(key)
		i+=1
	print  best




	gx=[]
	sx=[]
	lsx=[]
	nx1=[]
	nx2=[]
	nx3=[]
	rx=[]
	nx=[]
	bb=[]
	best_feature_no=[0.15,0.1,0.01,.001,.0001,.000001,0.0000000001,0.0]
	for i in best_feature_no:
		g,s,ls,n1,n2,n3,r,n,best2=cross_val2(i,ltr,lts,l,l2)
		gx.append(g)
		sx.append(s)
		lsx.append(ls)
		nx1.append(n1)
		nx2.append(n2)
		nx3.append(n3)
		rx.append(r)
		nx.append(n)
		bb.append(best2)




	print "bb: ",bb
	u=plt.figure(1)
	u.canvas.set_window_title('Feature Selection:  VarianceThreshold')
	plt.plot(bb, gx, label='GNB')
	plt.plot(bb, sx, label='SVC')
	plt.plot(bb, lsx, label='LinearSVC')
	plt.plot(bb, nx1, label='Neural Network.adam')
	plt.plot(bb, nx2, label='Neural Network.lbfgs')
	plt.plot(bb, nx3, label='Neural Network.sgd')
	plt.plot(bb, rx, label='RandomForest')
	plt.plot(bb, nx, label='Nearest Neighbors')

# Add a legend
	plt.legend()
	plt.xlabel('Feature_set_size', fontsize=18)
	plt.ylabel('Accuracy', fontsize=16)
# Show the plot


	gx = []
	sx = []
	lsx = []
	nx1 = []
	nx2 = []
	nx3 = []
	rx = []
	nx = []
	best_feature_no =bb
	for i in best_feature_no:
		g, s, ls, n1, n2, n3, r, n = cross_val(i, ltr, lts, l, l2)
		gx.append(g)
		sx.append(s)
		lsx.append(ls)
		nx1.append(n1)
		nx2.append(n2)
		nx3.append(n3)
		rx.append(r)
		nx.append(n)

	v=plt.figure(2)
	v.canvas.set_window_title('Feature Selection: SelectKBest with ANOVA')
	plt.plot(best_feature_no, gx, label='GNB')
	plt.plot(best_feature_no, sx, label='SVC')
	plt.plot(best_feature_no, lsx, label='LinearSVC')
	plt.plot(best_feature_no, nx1, label='Neural Network.adam')
	plt.plot(best_feature_no, nx2, label='Neural Network.lbfgs')
	plt.plot(best_feature_no, nx3, label='Neural Network.sgd')
	plt.plot(best_feature_no, rx, label='RandomForest')
	plt.plot(best_feature_no, nx, label='Nearest Neighbors')

	# Add a legend
	plt.legend()
	plt.xlabel('Feature_set_size', fontsize=18)
	plt.ylabel('Accuracy', fontsize=16)
	# Show the plot




	plt.show()

	p = SelectKBest(f_classif, k=22)
	ltrn = p.fit_transform(ltr, l)
	ltsn = p.transform(lts)
	print "Feature Scores: ",ltrn.shape



	map={}
	for i in range(len(feature_name)):
		#print feature_name[i]," = ",p.scores_[i]
		map[feature_name[i]]=p.scores_[i]

	kk=[]
	for key, value in sorted(map.iteritems(), key=lambda (k, v): (v, k)):
		print key," ",value
		kk.append(key)
	bestt=[]
	i=1
	for key in reversed(kk):
		if i==bbb+1:
			break
		bestt.append(key)
		i+=1
	print  "Variance Threshold: ",best," ",len(best)
	print  "ANOVA:              ",bestt, len(bestt)

	bbest=set(best)


	for i in bestt:
		bbest.add(i)

	print bbest
	print set(best).intersection(bestt)
