import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#open_=pd.read_csv("winequality-white.csv")
open_=pd.read_csv("winequality-red.csv")
x=pd.DataFrame(open_[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]).to_numpy()
y=pd.DataFrame(open_['quality']).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(x, y)
print(np.size(x))

#plt.scatter(x,y)
#plt.show()

model=RandomForestClassifier().fit(X_train,y_train)
pred = model.predict(X_test)
for itr in range(0,10):



    print("real {} : predicted: {}".format(y_test[itr],pred[itr]))


