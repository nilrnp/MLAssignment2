#-------------------------------------------------------------------------
# AUTHOR: Nil Patel
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program has access to training data. The program then tests itself compared to the training data.
# FOR: CS 4210- Assignment #2
# TIME SPENT: about 1 and a half hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    features = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3, 'Myope': 1, 'Hypermetrope': 2, 'Yes': 1,
                         'No': 2, 'Normal': 1, 'Reduced': 2}
    numOfAttributes = 4
    for row in range(len(dbTraining)):
        tmp = []
        for feature in range(0, numOfAttributes):
            tmp.append(int(features[dbTraining[row][feature]]))
        X.append(tmp)


    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    for row in range(len(dbTraining)):
        Y.append(int(features[dbTraining[row][4]]))

    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       # dbTest =
       dbTestBefore = []
       with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for index, row in enumerate(reader):
               if index > 0:  # skipping the header
                   dbTestBefore.append(row)

       dbTest = []
       for row in range(len(dbTestBefore)):
           temp = []
           for feature in range(0, numOfAttributes + 1):
               temp.append(int(features[dbTestBefore[row][feature]]))
           dbTest.append(temp)
       numCorrect = 0
       resultCount = 0
       
       for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           predictedClass = clf.predict([[data[0], data[1], data[2], data[3]]])[0]

           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
           if predictedClass == data[4]:
              numCorrect = numCorrect + 1
           resultCount = resultCount + 1
           
    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
       accuracy = numCorrect / resultCount
       worstAccuracy = 1
       if (accuracy < worstAccuracy):
           worstAccuracy = accuracy
           
    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
print("final accuracy when training on " + ds + ": " + str(worstAccuracy))
