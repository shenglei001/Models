'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score #模型检验指标


import numpy as np
f=open('C:\\Users\\Shenglei\\Desktop\\Python\\rmc_inher_fmspread_in.csv','r')

sens=f.readlines()
f.close()

n=0
datalst=[]
for sen in sens:

    if n==0:
        n+=1
        continue
    else:
        n+=1
        
    senlst = sen.strip().split(',')
    senlst = senlst[1:]
    senlst = [float(x) for x in senlst]
    datalst.append(senlst)

data=np.array(datalst)



X = data[:,:5]
#print(X[:10])

Y = data[:,-1]

#print(Y[:10])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y)

print(X_train[:10,])
print(y_train[:10,])



print(X_test[:10,])
print(y_test[:10,])







from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
("kmeans", KMeans(n_clusters=10)),
("log_reg", LinearRegression()),
])
pipeline.fit(X_train, y_train)
print('model score:',pipeline.score(X_test, y_test))


'''
##
##### 多模型的对比
####reg=linear_model.LinearRegression() #建立线性回归模型
####reg.fit (X,Y) #训练模型 .fit(x_train,y_train)
####predict_Y=reg.predict(X)
####print('Variance score: ' ,reg.score(X,Y)) # 模型打分: 越接近1越好
####print(reg.coef_)
####print(reg.intercept_)
##
##
##
##def which_reg_better(reg):
##    reg.fit(X,Y) #训练模型
##    predict_Y=reg.predict(X)
##    print("Variance score:",r2_score(Y,predict_Y))
##    print(reg.coef_)
##    print(reg.intercept_)
##
##
####
####X=load_data.data
####Y=load_data.target
##
###x2=[2.569999933,0.932970345,873.3439941,5.634033]
###x2=[3.5,1031.526978,0.913333595,875.6743774,1035.897095]
###x2=[3,1280.440186,0.929230392,886.1299438,1290.537354]
##x2=[2,1227.912842,0.945234597,865.9650879,1230.973022]
##
##
##
##
##
##x2=np.array(x2)
##
##
##reg=linear_model.LinearRegression() #建立模型
##print('线性回归：')
##which_reg_better(reg)
##print('predict',reg.predict([x2[:5]]))
##
##
##reg=linear_model.Ridge() #建立模型
##print('岭回归：')
##which_reg_better(reg)
##print('predict',reg.predict([x2[:5]]))
##
##reg=linear_model.Lasso() #建立模型
##print('Lasson回归：')
##which_reg_better(reg)
##print('predict',reg.predict([x2[:5]]))
##
##reg=linear_model.BayesianRidge() #建立模型
##print('贝叶斯回归：')
##which_reg_better(reg)
##print('predict',reg.predict([x2[:5]]))
##

###方法2，多元线性回归

import csv
import numpy as np

def readData():
    X = []
    y = []
    with open('C:\\Users\\Shenglei\\Desktop\\Python\\rmc_inher_fmspread_in.csv') as f:
        rdr = csv.reader(f)
        # Skip the header row
        next(rdr)
        # Read X and y
        for line in rdr:
            xline = [] #xline = [1.0]
            #print(xline)
            for s in line[1:-2]:
                xline.append(float(s))
            X.append(xline)
            y.append(float(line[-1]))
    return (X,y)

X0,y0 = readData()
##print(X0[:10])
# Convert all but the last 10 rows of the raw data to numpy arrays
d = len(X0)-10
X = np.array(X0[:d])
y = np.transpose(np.array([y0[:d]]))

# Compute beta
Xt = np.transpose(X)
XtX = np.dot(Xt,X)
Xty = np.dot(Xt,y)
beta = np.linalg.solve(XtX,Xty)
print(beta)

# Make predictions for the last 10 rows in the data set
for data,actual in zip(X0[d:],y0[d:]):
    x = np.array([data])
    prediction = np.dot(x,beta)
    print('prediction = '+str(prediction[0,0])+' actual = '+str(actual))









