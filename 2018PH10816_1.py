#!/usr/bin/env python
# coding: utf-8

# In[4]:


### submission version
#with tanh 

import matplotlib.pyplot as plt
import numpy as np
import csv
import random 

###### Note : testing has to be defined at the  beginning don't change it in middle 
# testing true runs the version with train and val data and false writes output to csv 
testing = True
def digit_to_vector(d):
    v = np.zeros([10,1])
    v[int(d)] = 1
    return v







if testing == False :
    # opening training data
    file1 = 'mnist_train.csv' 
    raw_data = open(file1, 'rt')
    reader1 = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    read1 = list(reader1)
    data1 = np.array(read1).astype('float')
    print(data1.shape)
    p1 = data1[:,1:]
    q1 = data1[:,0]
    training_inputs = [np.reshape(i,(784,1)) for i in p1]
    training_results = [digit_to_vector(j) for j in q1]
    training_data = list(zip(training_inputs , training_results))
    # opening testing data 
    file2 = 'mnist_test.csv' 
    raw_data = open(file2, 'rt')
    reader2 = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    read2 = list(reader2)
    data2 = np.array(read2).astype('float')
    print(data2.shape)
    p = data2
    test_data = [np.reshape(i,(784,1)) for i in p]
elif testing == True:
    # splitting 7000 into 4900 training and 2100 test (validation)
    file1 = 'mnist_train.csv' 
    raw_data = open(file1, 'rt')
    reader1 = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    read1 = list(reader1)
    data1 = np.array(read1).astype('float')
    print(data1.shape)
    p1 = data1[:,1:]
    q1 = data1[:,0]
    inputs = [np.reshape(i,(784,1)) for i in p1]
    training_inputs = inputs[:4900]
    validation_inputs = inputs[4900:]
    training_results = [digit_to_vector(j) for j in q1[:4900]]
    validation_results = q1[4900:]
    training_data = list(zip(training_inputs , training_results))
    validation_data = list(zip(validation_inputs , validation_results))   

structure = [784,70,10]
# each item is a layer , and value of each item is the no. of neurons in that layer . 
numlayers = len(structure)
biases = [ np.random.randn(y,1) for y in structure[1:] ]
weights = [ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(structure[:-1],structure[1:])]


# to normalise tanh in 0 to 1 
#normalized = (x-min(x))/(max(x)-min(x))
def normalise(a):
    return (a + 1)/2
def feedf(a,weights,biases,act): 
    for b, w in zip(biases,weights):
      a = sigmoid(np.dot(w, a)+b)
    if( act == "tanh"):
        a = normalise(a)
    return a 
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z) + np.exp(-z))
def deltanh(z):
    return 1 - tanh(z)*tanh(z)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def delsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
def actfunction(z,act = "sig"):
    if  act == "sig" :
        return sigmoid(z)
    elif act == "tanh":
        return tanh(z)
    else :
        print("wrong activation name")
def delactfunction(z,act = "sig"):
    if  act == "sig" :
        return delsigmoid(z)
    elif act == "tanh":
        return deltanh(z)
    else :
        print("wrong activation name")
        
def evaluate(test_data,weights,biases,act):
    test_results = [(np.argmax(feedf(x,weights,biases,act)), y) for (x, y) in test_data] 
    total = sum(int(x == y) for (x, y) in test_results)
    n = len(test_data)
    return (total*100)/n

def backp(x, y,weights,biases,num_layers,act):
    btemp = [np.zeros(b.shape) for b in biases] 
    wtemp = [np.zeros(w.shape) for w in weights] 
    zArr = []
    aArr = []
    a = x
    aArr.append(a)
    
    for b,w in zip(biases,weights):
      z = np.dot(w,a) + b
      zArr.append(z)
      a = actfunction(z,act)
      aArr.append(a) 
    #delta = cost_derivative(aArr[-1], y) * delsigmoid(zArr[-1])
    # for cross entropy in last layer last layer
    delta = aArr[-1]-y
    btemp[-1] = delta 
    wtemp[-1] = np.dot(delta, aArr[-2].transpose())
    for l in range(2, num_layers):
       z = zArr[-l] 
       delta = np.dot(weights[-l+1].transpose(), delta) * delactfunction(z,act) 
       btemp[-l] = delta 
       wtemp[-l] = np.dot(delta, aArr[-l-1].transpose()) 
    return (btemp, wtemp)




def batch_update( batch, eta,weights,biases,lbda,tds,act):
    for x, y in batch:
       dbiases,dweights = backp(x, y,weights,biases,numlayers,act) 
    weights = [(1-eta*(lbda/tds))*w-(eta/len(batch))*dw for w, dw in zip(weights, dweights)] 
    biases = [b-(eta/len(batch))*db for b, db in zip(biases, dbiases)]
    return (weights,biases)




def Train(training_data, epochs, batch_size, eta, test_data, weights , biases,lbda,testing,act = "sig"):
    print("act is ")
    print(act)
    #print(testing)
    if testing == True :
       # print("testing")
        accuracies = []
        epocharr = []
    tds = len((training_data))
    n = len((training_data)) 
    for epoch in range(epochs):
      if testing == True :
        #print("epoch arr updated")
        epocharr.append(epoch+1)
      random.shuffle(training_data) 
      batches = [ training_data[i:i+batch_size] for i in range(0, n, batch_size)] 
      for batch in batches: 
        temp = batch_update(batch, eta,weights,biases,lbda,tds,act = "sig")
        weights = temp[0]
        biases = temp[1]
      if testing == True :
        print ("Epoch = {} , accuracy = {} % ".format( str(epoch+1), str(evaluate(test_data,weights,biases,act))))
        accuracies.append(evaluate(test_data,weights,biases,act))
      else:
        print( "Epoch = {} ".format(epoch+1))
        
    if testing == True:
       # print("returning accuracies and epocharr")
        return (accuracies,epocharr)
    else :
        return (weights,biases)
        
        

  
if testing == False:
    temp = Train(training_data, 300, 10, 0.003, test_data,weights,biases,7,False,"tanh")
    weights = temp[0]
    biases = temp[1]
    test_results = [np.argmax( feedf(x,weights,biases,"tanh") ) for x in test_data]
    fields = ['id', 'label']
    submitfile = 'submit.csv'
    with open(submitfile, 'w',newline='') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerow(fields) 
        #i = 1
        rows1 = [i for i in range(1,1+len(test_results))]
        arr1 = np.array(rows1)
        arr1 = arr1.reshape(len(arr1),1)
        rows2 = [  x for x in test_results]
        arr2 = np.array(rows2)
        arr2 = arr2.reshape(len(arr2),1)
        arr = np.hstack((arr1, arr2))
        # writing the data rows 
        csvwriter.writerows(arr)
    
else :
   
    '''
    print(structure)

    batchsizes = [1,5,10,50,100,200,500,1000,2450,4900]
    print( "Epochs = {}  , eta = {} , lambda = {} ,batchsize ".format(100,0.005,7) )
    for batchsize in batchsizes:
        biases = [ np.random.randn(y,1) for y in structure[1:] ]
        weights = [ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(structure[:-1],structure[1:])]
        print("batchsize = {}".format(batchsize))
        temp = Train(training_data,100,batchsize,0.005,validation_data,weights,biases,7,True)
        accuracies = temp[0]
        epocharr = temp[1]
        plt.plot(epocharr,accuracies,label = batchsize)
    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy(%)")
    plt.title("Varying batchsize")
    plt.legend()
    plt.show()
    '''
    structure = [784,70,10]
    print("Comparing sigmoid and tanh")
    print( "Epochs = {}  , eta = {} , lambda = {} ,batchsize = {} ".format(100,0.005,7,10) )

    print("using sigmoid")
    biases = [ np.random.randn(y,1) for y in structure[1:] ]
    weights = [ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(structure[:-1],structure[1:])]
    temp = Train(training_data,100,10,0.005,validation_data,weights,biases,7,True)
    accuracies = temp[0]
    epocharr = temp[1]
    plt.plot(epocharr,accuracies,label = "sigmoid")


    print("using tanh")
    biases = [ np.random.randn(y,1) for y in structure[1:] ]
    weights = [ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(structure[:-1],structure[1:])]
    temp = Train(training_data,100,10,0.005,validation_data,weights,biases,7,True,"tanh")
    accuracies = temp[0]
    epocharr = temp[1]
    plt.plot(epocharr,accuracies,label = "tanh")


    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy(%)")
    plt.title("Comparing sigmoid and tanh")
    plt.legend()
    plt.show()

    '''
    print("varying hidden layers")


    structures = [    [784,70,10] ,[784,70,70,10] ,[784,70,70,70,10],]
    print( "Epochs = {}  , eta = {} , lambda = {} ,batchsize = {} ".format(100,0.005,300,10) )
    u = 1
    for arr in structures :
        structure = arr
        biases = [ np.random.randn(y,1) for y in structure[1:] ]
        weights = [ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(structure[:-1],structure[1:])]
        print("No. of hidden layers = {}".format(u))
        temp = Train(training_data,100,10,0.005,validation_data,weights,biases,300,True)
        accuracies = temp[0]
        epocharr = temp[1]
        plt.plot(epocharr,accuracies,label = u)
        u = u + 1
    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy(%)")
    plt.title("Varying hidden layers")
    plt.legend()
    plt.show()
  
'''



'''
    print("varying hidden layers")


    structures = [ [784,70,10] ,[784,70,70,10] ,[784,70,70,70,10]]
    print( "Epochs = {}  , eta = {} , lambda = {} ,batchsize = {} ".format(100,0.005,7,10) )

    for arr in structures :
        structure = arr
        biases = [ np.random.randn(y,1) for y in structure[1:] ]
        weights = [ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(structure[:-1],structure[1:])]
        print("No. of neurons in hidden layer = {}".format(len(arr)-2)
        temp = Train(training_data,100,10,0.005,validation_data,weights,biases,7,True)
        accuracies = temp[0]
        epocharr = temp[1]
        plt.plot( epocharr,accuracies,label = len(arr)-2 )
    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy(%)")
    plt.title("Varying no. of neurons of single hidden layer")
    plt.legend()
    plt.show()
  
 

'''
















    
    
    
    
'''
    lmdas = [0.1,1,5,10,20,50,100]
    print( "Epochs = {} , batchsize = {} , eta = {} , varying lambda ".format(100,10,0.005) )
    for lmda in lmdas:
        biases = [ np.random.randn(y,1) for y in structure[1:] ]
        weights = [ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(structure[:-1],structure[1:])]
        print("lamba = {}".format(lmda))
        temp = Train(training_data,100,10,0.005,validation_data,weights,biases,lmda,True)
        accuracies = temp[0]
        epocharr = temp[1]
        plt.plot(epocharr,accuracies,label = lmda)
    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.show()
'''

'''

    print( "Epochs = {} , batchsize = {} , lambda = {} , varying eta ".format(100,10,5) )
    eta = 0.001
    while eta<5 :
     
        print( "eta = {}".format(eta))
    
        biases = [ np.random.randn(y,1) for y in structure[1:] ]
        weights = [ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(structure[:-1],structure[1:])]
        temp = Train(training_data, 100, 10, eta, validation_data,weights,biases,5,True)
        accuracies = temp[0]
        epocharr = temp[1]
        plt.plot(epocharr,accuracies,label = eta)  
        #plt.xlabel("No. of Epochs")
        #plt.ylabel("Accuracy(%)")
        #plt.legend()
        #plt.show()
        eta = eta* 3
    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.show()
''' 
    







    
        


# In[ ]:




