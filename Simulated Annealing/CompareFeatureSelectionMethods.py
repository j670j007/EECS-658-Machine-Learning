#EECS 658 Assignment4
#CompareFeatureSelectionMethods.py
#This program creates a series confusion matrices for a series of machine learning models the iris.csv dataset. 
#The ML Models are as follows: Decision Tree, PCA, Simulated Annealing, and Genetic Algorithm 
#After each model, accuracy, confusion matrix, and list of considered features are printed out.   
#inputs: iris.csv
#outputs: confusion matrix, accuracy score, features for each test
#most of the code below was included in the course slides. I consulted the
#Numpy User Guide https://docs.scipy.org/doc/numpy/user/index.html for more information on 
#the np functions
#I consulted https://scikit-learn.org/ for information on each of the ML Models functions
#For additional information about array manipulation in Python, I consulted:
#https://problemsolvingwithpython.com/05-NumPy-and-Arrays/05.06-Array-Slicing/
#Jennifer Aber
#October 12, 2023
# load libraries
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing   
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import random
import math

#load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length',
'petal-width', 'class']
dataset = read_csv(url, names=names)



  
# Use label_encoder object to convert 
# Names to numeric values
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'class', changes text values to numbers 
dataset['class'] = label_encoder.fit_transform(dataset['class'])
  

#Create Arrays for Features and Classes
array = dataset.values
X = array[:,0:4] #array X with flower features (petal length, etc..) - all five columns
y = array[:,4] #array y with flower names - col. 5
#Split Data into 2 Folds for Training and Test using sklearn functionality
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y,
test_size=0.50, random_state=1)
                

#creates a model of the data using DecisionTreeClassifier()
tree = DecisionTreeClassifier()
tree.fit(X_Fold1, y_Fold1) #first fold training
treepred1 = tree.predict(X_Fold2) #first fold testing
tree.fit(X_Fold2, y_Fold2) #second fold training
treepred2 = tree.predict(X_Fold1) #second fold testing

actual = np.concatenate([y_Fold2, y_Fold1]) #combines actual classes from both partitions
treepredicted = np.concatenate([treepred1, treepred2]) #combines predicted classes from both partitions

print( 'Decision Tree') #name of regression model
print('Accuracy Score: ' + str(round(accuracy_score(actual, treepredicted), 3) ))
print('Confusion Matrix: ' )
print(confusion_matrix(actual, treepredicted)) #confusion matrix

print("")
print( 'PCA Feature Reduction') # Start

# Create PCA instance
pca = PCA(n_components=4)

# Perform PCA
pca.fit(X)

# Get eigenvectors and eigenvalues
eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

print("EigenVectors :")
print(eigenvectors)
print("Eigenvalues :")
print(eigenvalues)


# Transform data
principleComponents = pca.transform(X)
NamesArray = pca.get_feature_names_out()

# Calculate PoVs
sumvariance = np.cumsum(eigenvalues)
sumvariance /= sumvariance[-1]

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = list(zip(eigenvalues, eigenvectors))

# Sort the (eigenvalue, eigenvector) tuples from high to low
# eigen_pairs.sort(key=lambda x: x[0], reverse=True)

# Transform data (x) to Z Using First Eigenvector
W = eigen_pairs[0][1].reshape(4, 1)
Z = principleComponents.dot(W)

#Split PCA Data (Z) into 2 Folds for Training and Test using sklearn functionality
PCA_Fold1, PCA_Fold2, y_Fold1, y_Fold2 = train_test_split(Z, y,
test_size=0.50, random_state=0) # Use Random_state = 0 for consistent results

#creates a model of the PCA data using DecisionTreeClassifier()
tree = DecisionTreeClassifier()
tree.fit(PCA_Fold1, y_Fold1) #first fold training
treepred1 = tree.predict(PCA_Fold2) #first fold testing
tree.fit(PCA_Fold2, y_Fold2) #second fold training
treepred2 = tree.predict(PCA_Fold1) #second fold testing

actual = np.concatenate([y_Fold2, y_Fold1]) #combines actual classes from both partitions
treepredicted = np.concatenate([treepred1, treepred2]) #combines predicted classes from both partitions

print( 'Decision Tree after PCA') #Results After PCA
print('Confusion Matrix: ' )
print(confusion_matrix(actual, treepredicted)) #confusion matrix
print('Accuracy Score: ' + str(round(accuracy_score(actual, treepredicted), 3) ))
print("Feature Name = ", NamesArray[0])
print("PoV = ", sumvariance[0])

print("Done")


#Simulated Annealing 
W2= eigenvectors
Z2=principleComponents.dot(W2)


#creating an 8-column array with the results of PCA and the original iris dataset 

combinedArray = np.concatenate([X, Z2], axis=1)

#all 8 feature names

mySelection = np.array([1,1,1,1,1,1,1,1])
names1 = names[:4]
names2 = np.array(['Z1', 'Z2', 'Z3', 'Z4'])
#print(names1)

combined_names = np.concatenate([names1, names2])
#print(combined_names)

#adds or removes features at random

def perturb(oldarray):
    newarray = oldarray
    # let 0 = add and 1 = delete
    add_remove = random.choice([0, 1])
    # 1 = remove 1 and 2 = remove 2
    howmany = random.choice([1, 2])
    if howmany > np.count_nonzero(newarray==1) - 1:
        add_remove = 0
    elif howmany > np.count_nonzero(newarray==0):
        add_remove = 1

    if add_remove == 0:
        newarray = add(newarray)
        if howmany == 2:
            newarray = add(newarray)
    if add_remove == 1:
        newarray = remove(newarray)
        if howmany == 2:
            newarray = remove(newarray)
        
    return newarray


#remove subroutine, randomly selects which feature to remove
def remove(oldarray): 
    newarray = oldarray      
    which = random.randrange(1, np.count_nonzero(newarray==1) + 1)  #np.count_nonzero checks to make sure there's at least 1 '1' in the array
    count=0
    for i in range (0, 8):
        if newarray[i] == 1:
            count= count+1
            if count == which:
                newarray[i] = 0
    return newarray

#remove subroutine, randomly selects which feature to remove
def add(oldarray):
    newarray = oldarray
    if np.count_nonzero(newarray==0) == 1:
        for i in range (0, 8):
            if newarray[i] == 0:
                newarray[i] = 1
    else:
        which = random.randrange(1, np.count_nonzero(newarray==0))
        count=0
        for i in range (0, 8):
            if newarray[i] == 0:
                count= count+1
                if count == which:
                    newarray[i] = 1
    return newarray

def main_loop():
    cur_array = mySelection   #array of 8 features
    restart_counter = 0
    feature_subset = mySelection
    best_accuracy = 0
    best_feature_subset = feature_subset
    old_acc = 0
    for i in range(1, 101):
        cur_array,newaccuracy = SA(cur_array, i, old_acc)  #calls Simulated Annealing subroutine with array, index, and previous accuracy
        old_acc = newaccuracy
        if newaccuracy > best_accuracy: #tracks best accuracy
            best_accuracy = newaccuracy
            best_feature_subset = cur_array  #tracks best set of features
            restart_counter = 0
        else:
            restart_counter = restart_counter + 1
            if restart_counter == 10:
                feature_subset = best_feature_subset
                restart_counter = 0
       
        
        print_features(cur_array)

    print("Best Accuracy:", best_accuracy)
    print("Best Feature Subset:")
    print_features(best_feature_subset)  #prints the best feature names
    best = build_subsets(best_feature_subset)  #ties arrays of '0' and '1' to actual datasets
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(best, y, test_size=0.50, random_state=1)  #2-fold cross-validation - same as above 
    SAtree = DecisionTreeClassifier()
    SAtree.fit(X_Fold1, y_Fold1) #first fold training
    SAtreepred1 = SAtree.predict(X_Fold2) #first fold testing
    SAtree.fit(X_Fold2, y_Fold2) #second fold training
    SAtreepred2 = SAtree.predict(X_Fold1) #second fold testing

    actual = np.concatenate([y_Fold2, y_Fold1]) #combines actual classes from both partitions
    SAtreepredicted = np.concatenate([SAtreepred1, SAtreepred2]) #combines predicted classes from both partitions
    #getting the confusion matrix for the best set of features
    print('Confusion Matrix: ' )
    print(confusion_matrix(actual, SAtreepredicted))
        

def SA(oldarray, i, prev_acc):   #simulated annealing algorithm
    newarray= oldarray
    iteration = i
    previous = prev_acc
    newarray = perturb(oldarray)
    accuracy = fit_estimate(newarray)
    if accuracy > previous:
       print(i, np.count_nonzero(newarray==1), round(accuracy, 3), "---", "---", "Improved") 
    else:
        #Calculate acceptance probabillty;  Pr[accept] = e^((-i/c)((old-new)/old))
        pr_accept = math.exp(-iteration*((previous - accuracy)/previous))
        #Generate Random Uniform Variable 
        r = random.random()
        #if random uniform variable > probability then
        if r > pr_accept:
            #Reject new subset
            print(i, np.count_nonzero(newarray==1), round(accuracy, 3), round(pr_accept, 3), round(r, 3), "Reject") 
        else:
            #Accept new subset
            print(i, np.count_nonzero(newarray==1), round(accuracy, 3), round(pr_accept, 3), round(r, 3), "Accept") 
    return newarray,accuracy

def print_features(oldarray):  #subroutine to get the feature names
    newarray=oldarray
    for i in range(0, 8):   
        if newarray[i] == 1:
            print(combined_names[i])


def build_subsets(oldarray):
    #This subroutine takes the array of '1' and '0' and ties it to the data, adding selected features
    mySelection = oldarray
    subset = combinedArray[:,0]  #  extra column, trimmed below
    #length = Subset.size 
    subset = np.reshape(subset,[subset.size,1])
    for i in range(0, mySelection.size):
        if mySelection[i] == 1:  #separates 8-column array of data into individual columns, adding select ones
            add_on = combinedArray[:,i]
            add_on = np.reshape(add_on,[add_on.size,1])
            subset = np.concatenate([subset,add_on], axis=1)
    subset = np.delete(subset, 0, 1)  #trimming the extra column 
    return subset

def fit_estimate(oldarray):
    subset = build_subsets(oldarray)
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(subset, y, test_size=0.50, random_state=1)
    SA_model = DecisionTreeClassifier()
    SA_model.fit(X_Fold1, y_Fold1) #first fold training
    SApred1 = SA_model.predict(X_Fold2) #first fold testing
    SA_model.fit(X_Fold2, y_Fold2) #second fold training
    SApred2 = SA_model.predict(X_Fold1) #second fold testing
    actual = np.concatenate([y_Fold2, y_Fold1]) #combines actual classes from both partitions
    SApredicted = np.concatenate([SApred1, SApred2]) #combines predicted classes from both partitions
    accuracy = accuracy_score(actual, SApredicted)
    return accuracy

print("Simulated Annealing")
main_loop()   #runs Simulated Annealing 

def initialize():  # Returns 5 specified individuals
    one = np.array([1,1,1,1,1,0,0,0]) # 
    two = np.array([0,1,1,1,1,1,0,0]) #
    three = np.array([0,1,1,0,1,1,1,0]) #
    four = np.array([0,1,0,0,1,1,1,1]) #
    five = np.array([1,0,0,0,1,1,1,1]) #

    #builds an array of 8 columns, five rows to represent sets of features
    ancestors = np.concatenate([one, two])
    ancestors = np.concatenate([ancestors, three])
    ancestors = np.concatenate([ancestors, four])
    ancestors = np.concatenate([ancestors, five])

    ancestors = np.reshape(ancestors, (5, 8))
    return ancestors

#uses exclusive 'OR' to return unions of the sets
def unions(mySet): 
    newunions = mySet

    result1 = OR(newunions[0], newunions[1])
    result2 = OR(newunions[0], newunions[2])
    result3 = OR(newunions[0], newunions[3])
    result4 = OR(newunions[0], newunions[4])
    result5 = OR(newunions[1], newunions[2])
    result6 = OR(newunions[1], newunions[3])
    result7 = OR(newunions[1], newunions[4])
    result8 = OR(newunions[2], newunions[3])
    result9 = OR(newunions[2], newunions[4])
    result10 = OR(newunions[3], newunions[4])

    
    newunions = np.concatenate([result1, result2])
    newunions = np.concatenate([newunions, result3])
    newunions = np.concatenate([newunions, result4])
    newunions = np.concatenate([newunions, result5])
    newunions = np.concatenate([newunions, result6])
    newunions = np.concatenate([newunions, result7])
    newunions = np.concatenate([newunions, result8])
    newunions = np.concatenate([newunions, result9])
    newunions = np.concatenate([newunions, result10])
    newunions = np.reshape(newunions, (10, 8))
    
    #print(newunions)
    return newunions

#uses 'AND' to return intersections of the sets
def intersection(mySet): 
    newintersections = mySet

    result1 = AND(newintersections[0], newintersections[1])
    result2 = AND(newintersections[0], newintersections[2])
    result3 = AND(newintersections[0], newintersections[3])
    result4 = AND(newintersections[0], newintersections[4])
    result5 = AND(newintersections[1], newintersections[2])
    result6 = AND(newintersections[1], newintersections[3])
    result7 = AND(newintersections[1], newintersections[4])
    result8 = AND(newintersections[2], newintersections[3])
    result9 = AND(newintersections[2], newintersections[4])
    result10 = AND(newintersections[3], newintersections[4])

    
    newintersections = np.concatenate([result1, result2])
    newintersections = np.concatenate([newintersections, result3])
    newintersections = np.concatenate([newintersections, result4])
    newintersections = np.concatenate([newintersections, result5])
    newintersections = np.concatenate([newintersections, result6])
    newintersections = np.concatenate([newintersections, result7])
    newintersections = np.concatenate([newintersections, result8])
    newintersections = np.concatenate([newintersections, result9])
    newintersections = np.concatenate([newintersections, result10])
    newintersections = np.reshape(newintersections, (10, 8))
    
    #print(newintersections)
    return newintersections

#XOR
def OR(a, b):
    arr1 = a
    arr2 = b

    result = a|b
    return result
#AND
def AND(a, b):
    arr1 = a
    arr2 = b

    result = a&b
    return result

#adds or removes feature at random.  Checking in place to make sure we have at least 1 '0' or 1 '1' to add to or remove, respectively
def mutate(mySet):
    mutated = mySet
    for i in range (0, 25):
        myrandom = (random.choice([0, 1]))
        if myrandom == 0:
            if np.count_nonzero(mutated[i]==0) > 0:
                mutated[i] = add(mutated[i])
            else:
                mutated[i] = remove(mutated[i])
        elif myrandom == 1:
            if np.count_nonzero(mutated[i]==1) > 1:
                mutated[i] = remove(mutated[i])
            else:
                mutated[i] = add(mutated[i])

    return mutated

def evaluate(mySet):
    set = mySet  #takes 50 * 8 array of features 
    accuracies = np.zeros((50,2))  #np.zeros function creates a 50*2 array of zeros we can overwrite with accuracies and indexes below
    best_features = best  #5 * 8 array - best sets of features.  Starts with original five for first iteration. 
    for i in range(0, 50): 
        current_subset = set[i]
        accuracy = fit_estimate(current_subset)
        accuracies[i] = accuracy, i   #loading up a 50 * 2 array for sorting purposes 
     
    inOrder = accuracies[np.argsort(accuracies[:,0])]  #sorting by accuracy 
    descending = inOrder[::-1] # reverse order

    for i in range(0,5): #getting the best 5 features
        print("I , Acc , Feat ", i, descending[i][0], descending[i][1])  #index, accuracy, # index # of the best sets of features 
        best_features[i] = set[int(descending[i][1])]

    print("-----------------------------------------------------------------")


    return best_features


    
best = initialize()

#building the first 50 * 8 set from 5 features.  Additional 10 are union, 10 are intersection, then all 25 are mutated
union = unions(best)
print(union.shape)
intersections = intersection(best)
total = np.concatenate([best, union])
total = np.concatenate([total, intersections])
mutated = mutate(total)
total = np.concatenate([total, mutated])

for i in range(1, 50):  #50 generations 
    print("Generation", i)
    best = evaluate(total)   #get the best features
    union = unions(best)  #take those 5 and build a new 50 rows
    intersections = intersection(best)
    total = np.concatenate([best, union])
    total = np.concatenate([total, intersections])
    mutated = mutate(total)
    total = np.concatenate([total, mutated])
    

print("Best Features : ")
#prints the top features of the most recent generation
print_features(best[0])
    
subset = build_subsets(best[0])
# print("GA : ", subset)
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(subset, y, test_size=0.50, random_state=1)
SA_model = DecisionTreeClassifier()
SA_model.fit(X_Fold1, y_Fold1) #first fold training
SApred1 = SA_model.predict(X_Fold2) #first fold testing
SA_model.fit(X_Fold2, y_Fold2) #second fold training
SApred2 = SA_model.predict(X_Fold1) #second fold testing
actual = np.concatenate([y_Fold2, y_Fold1]) #combines actual classes from both partitions
SApredicted = np.concatenate([SApred1, SApred2]) #combines predicted classes from both partitions
accuracy = accuracy_score(actual, SApredicted)
print("Best Accuracy : ", accuracy)

print('Confusion Matrix: ' )
print(confusion_matrix(actual, SApredicted)) #confusion matrix


print("Done")
        


        

        


    
    
    
    
    



        
        

    















        















