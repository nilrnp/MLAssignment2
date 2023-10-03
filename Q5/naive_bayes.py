#-------------------------------------------------------------------------
# AUTHOR: Nil Patel
# FILENAME: naive_bayes.py
# SPECIFICATION: This program has access to weather data. The program then tests tje weather data compared to training data.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

dataSets = ['weather_training.csv', 'weather_test.csv']

#reading the training data in a csv file
#--> add your Python code here
db = []
with open(dataSets[0], 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: 
         db.append(row)

features = {'Sunny' : 1, 'Overcast':2, 'Rain':3, 'Hot':1, 'Mild':2, 'Cool':3,
                     'High':1, 'Normal':2, 'Weak':1, 'Strong':2, 'Yes':1, 'No':2}

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
X=[]
for row in range(len(db)):
    X.append([int(features[db[row][1]]),
              int(features[db[row][2]]),
              int(features[db[row][3]]),
              int(features[db[row][4]])])

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
Y=[]
for row in range(len(db)):
    Y.append(int(features[db[row][5]]))
  
#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
testDB = []
with open(dataSets[1], 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: 
         testDB.append(row)
         
#printing the header os the solution
#--> add your Python code here
print ("Day".ljust(20) + "Outlook".ljust(20) + "Temperature".ljust(20)  + "Humidity".ljust(20)  + "Wind".ljust(20)  + "PlayTennis".ljust(20)  + "Confidence".ljust(20))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
testX = []
for row in range(len(testDB)):
    testX.append([int(features[testDB[row][1]]),
              int(features[testDB[row][2]]),
              int(features[testDB[row][3]]),
              int(features[testDB[row][4]])])

testY = []
for i in range(len(testX)):
    testY.append(clf.predict_proba([testX[i]])[0])

    if testY[i][0] > .75:
      output = 1
    else:
      output = 0
      
    if output == 1:
      confidence = testY[i][0]
    else:
      confidence = testY[i][1]
    
    if output == 1:
      outputString = "Yes"
    else:
      outputString = "No"
      
    if testY[i][0] >= 0.75 or testY[i][1] >= 0.75:
        print(str(testDB[i][0]).ljust(20)
              + str(testDB[i][1]).ljust(20)
              + str(testDB[i][2]).ljust(20)
              + str(testDB[i][3]).ljust(20)
              + str(testDB[i][4]).ljust(20)
              + outputString.ljust(20)
              + str(round(confidence, 2)).ljust(20))