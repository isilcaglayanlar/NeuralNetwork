# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:00:22 2020

@author: ISIL
"""

from preprocess import read_data, create_samples, split_data, read_vectors, read_vocab
import numpy as np
import datetime
import random
from scipy.special import expit

#OVERWIEV
"""
hidden_layer_size/2 ≈ input_layer_size + output_layer_size
                    ≈ (50+2)2
                    ≈ 100
input layer(50 nodes) --W--> hidden layer(100 nodes) --U--> output layer(2 nodes)

--TRAINING--

FORWARD PROPOGITON
z = Wx+b
a = f(z) [tanh]
y = (U.transpose)a + b2
h = g(y) [sigmoid function]

E = binary cross-entropy error 

BACK PROPOGATION
Gradients
∂E/∂W = ∂E/∂h*∂h/∂y*∂y/∂a*∂a/∂z*∂z/∂W
∂E/∂U = ∂E/∂h*∂h/∂y*∂y/∂U

Gradient Descent 
W <- W - η*(∂E/∂W)
U <- U - η*(∂E/∂U)
"""

# HYPERPARAMETERS
input_size = 50 #size of each word vector
output_size = 2 #number of classes
hidden_layer_size = 100
learning_rate = 0.003 #η
number_of_epochs = 100000
C = 3 #window size
path = "./data" 

hidden_layer = np.zeros((1, hidden_layer_size))
output_layer = np.zeros((1, output_size))

#bias of hidden layer (H x 1), filled with ones
b1 = np.ones((hidden_layer_size, 1))

#weight of a connection (H x Cn)
W = np.random.randn(hidden_layer_size, C*input_size)

#logistic regression weights(H x 1)
U = np.random.rand(hidden_layer_size, 1)

#functions for data isn't included in vocab.txt file
def isDate(word):
    
    isDate = True
    try:
      datetime.datetime.strptime(word, '%Y-%m-%d')
    
    except ValueError:
        isDate = False
    
    return isDate    

def get_date_vector(): #embedded layer for date format data
    return read_vectors(path,28192) #index is chosen by using random initialization

def get_random_vector(): #embedded layer for other format data
    
    random_vector = []
    for i in range(input_size):
        random_value = round(random.uniform(-2.000000, 2.000000), 6)
        random_vector.append(random_value)
        
    return random_vector
    
#function that converts strings into word vectors
def embedding_layer(samples):
    x_list = []
 
    for i in range(C):
        words = read_vocab(path) #index of the word in the file "vocab.txt"
        word = (samples[i]).lower()
        
        if(word == "¡s¿"): #beginning token 
            word = "<s>"
        elif(word == "¡/s¿"): #end token
            word = "</s>"
        
        
        if(word in words):
            index = words.index(word)
            vector = read_vectors(path,index)
            x_list += vector #concatenate window elements' vectors
        
        else:
            if(isDate(word)):
                x_list += get_date_vector()
            
            else:
                x_list += get_random_vector()
                
    #input_layer
    x = np.array(x_list) 
    x = np.reshape(x,(C*input_size,1)) #Cn x 1
    
    return x

#add a single 1 to an array
def add_one(array):
    
    length = len(array)   
    
    new_array = np.zeros((length+1,1))
    new_array[0] = 1
    for i in range(length):
       
        new_array[i+1] = array[i]
    
    #print(new_array)
    return new_array

#f(z) = tanh(z)
def activation_function(layer):
	return np.tanh(layer)
 
#f'(signal) = 1-(tanh(signal)^2)
def derivation_of_activation_function(signal):
	return 1.0 - (np.tanh(signal)**2)

#binary cross-entropy error
def loss_function(true_labels, probabilities):
    if(true_labels[0]==1):
        if(probabilities[0]==0):
            return 0
        else:
            return -np.log(probabilities[0])
    elif(true_labels[1]==1):
        if(probabilities[1]==0):
            return 0
        else:
            return -np.log(probabilities[1])

# the derivation should be with respect to the output 
# -((y/y')+((1-y)/(1-y')))
#∂E/∂h
def derivation_of_loss_function(true_labels, probabilities):
	#return -((true_labels[0]/probabilities[0])-((1-true_labels[0])/(1-probabilities[0])))
    if(true_labels[0]==1):
        if(probabilities[0]==0):
            return 0
        else:
            return -1/probabilities[0]
    elif(true_labels[1]==1):
        if(probabilities[1]==0):
            return 0
        else:
            return -1/probabilities[1]

#g(z) = 1/(1+e^(-z))
def sigmoid_function(z):
    return expit(z) #to avoid overflow expit is used

#g'(z) = g(z)*(1-g(z))
def derivation_of_sigmoid_function(z):
    return sigmoid_function(z)*(1-sigmoid_function(z))

#W <- W - η*(∂E/∂W)
#U <- U - η*(∂E/∂U)
def gradient_descent(theta,deriv):
    
    # U : theta [100x1] | deriv [100x1] | layers [150x1]
    # W : theta [100x150] | deriv [150x1] | layers [1x150]
    
    theta = theta - (learning_rate*deriv)
    return theta

#∂E/∂U
def gradient_U(loss_signals,a):
    
    E_h = loss_signals #∂E/∂h
   
    y = np.dot((np.transpose(U)),a)

    #∂h/∂y = g'(y) [1x1]
    h_y = derivation_of_sigmoid_function(y) 
    
    #∂y/∂U = a [100x1]
    y_U = a 
    
    #∂E/∂U = ∂E/∂h*∂h/∂y*∂y/∂U [100x1]
    return E_h*h_y*y_U

#∂E/∂W
def gradient_W(loss_signals,x,a):
    
    E_h = loss_signals #∂E/∂h
    
    y = np.dot((np.transpose(U)),a)
    
    #∂h/∂y = g'(y) [1x1]
    h_y = derivation_of_sigmoid_function(y) 
    
    #∂y/∂a = U [1x100]
    y_a = U
    
    z = np.add((W.dot(x)),b1) # n x 1 = (n x Cn).(Cn x 1) + (n x 1)
    
    #∂a/∂z = f'(z) [1x100]
    a_z = derivation_of_activation_function(z)
    
    #∂z/∂W = x.transpose [1x150]
    z_W = np.transpose(x)
    
    #∂E/∂W = ∂E/∂h*∂h/∂y*∂y/∂a*∂a/∂z*∂z/∂W [100x150]
    
    #∂E/∂y = ∂E/∂h*∂h/∂y [scalar]
    E_y = E_h*h_y
    
    #∂y/∂W = ∂y/∂a*∂a/∂z*∂z/∂W [100x150]
    y_W = np.dot(np.multiply(y_a,a_z),z_W)
       
    #∂E/∂W = ∂E/∂y/∂y/∂W [100x150]
    return np.multiply(E_y,y_W)

#z = Wx+b
       
def forward_pass(data,U,W):
	
    predictions = []
    
    x = embedding_layer(data) #Cn x 1
    U = add_one(U) #including the bias term b(2) inside U
    
    #z = Wx+b
    z = np.add((W.dot(x)),b1) # n x 1 = (n x Cn).(Cn x 1) + (n x 1)
    #a = f(z) [tanh]
    a = activation_function(z) # n x 1
    a_1 = add_one(a) # (n+1) x 1
    #h = g((U.transpose)a + b2) [sigmoid function]    
    h = sigmoid_function(np.dot((np.transpose(U)),a_1)) # 1 x 1
        
    predictions.append(h[0][0])
    predictions.append(1-h[0][0])
   
    return predictions, a
    
def backward_pass(input_layer, hidden_layers , output_layer, loss_signals,U,W): 
    x = embedding_layer(input_layer)
    #∂E/∂U = ∂E/∂h*∂h/∂y*∂y/∂U
    deriv_U = gradient_U(loss_signals,hidden_layers)
    #∂E/∂W = ∂E/∂h*∂h/∂y*∂y/∂a*∂a/∂z*∂z/∂W
    deriv_W = gradient_W(loss_signals,x,hidden_layers)
    
    theta_U = U
    theta_W = W
    
    #Gradient Descent
    #U <- U - η*(∂E/∂U)
    theta_U = gradient_descent(theta_U,deriv_U) #[100x1]
    #W <- W - η*(∂E/∂W)
    theta_W = gradient_descent(theta_W, deriv_W) #[1x150]
    
    return theta_U, theta_W

def train(train_data, train_labels, valid_data, valid_labels):    
    
    theta_U = U
    theta_W = W
    
    for epoch in range(number_of_epochs):
        index = 0
        
        #for each batch
        for data, labels in zip(train_data, train_labels):
           predictions, hidden_layers = forward_pass(data,theta_U,theta_W)
           loss_signals = derivation_of_loss_function(labels, predictions)
           theta_U, theta_W = backward_pass(data, hidden_layers, predictions, loss_signals,theta_U,theta_W)
           loss = loss_function(labels, predictions)
         
           
           if index%20000 == 0: # at each 20000th sample, we run validation set to see our model's improvements
               accuracy, loss = test(valid_data, valid_labels,theta_U,theta_W)
               print("Epoch= "+str(epoch)+", Coverage= %"+ str(100*(index/len(train_data))) + ", Accuracy= "+ str(accuracy) + ", Loss= " + str(loss))
           
           index += 1
           
def test(test_data, test_labels,U,W):
    
    avg_loss = 0
    predictions = []
    labels = []
    
    theta_U = U
    theta_W = W
    
    #for each batch
    
    for data, label in zip(test_data, test_labels):
        prediction, hidden_layers = forward_pass(data,theta_U,theta_W)
        predictions.append(prediction)
        labels.append(label)
        avg_loss += np.sum(loss_function(label, prediction))
        
	#turn predictions into one-hot encoded 

    one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
    for i in range(len(predictions)):
        one_hot_predictions[i][np.argmax(predictions[i])] = 1
        
    predictions = one_hot_predictions
    accuracy_score = accuracy(labels, predictions)
    
    return accuracy_score,  avg_loss / len(test_data)


def accuracy(true_labels, predictions):
	true_pred = 0

	for i in range(len(predictions)):
		if np.argmax(predictions[i]) == np.argmax(true_labels[i]): # if 1 is in same index with ground truth
			true_pred += 1

	return true_pred / len(predictions)


if __name__ == "__main__":

    
	#PROCESS THE DATA
    words, labels = read_data(path)
    sentences = create_samples(words, labels)
    
    train_x, train_y, test_x, test_y = split_data(sentences)


	# creating one-hot vector notation of labels. (Labels are given numeric)
	# [0 1] is PERSON
	# [1 0] is not PERSON
    new_train_y = np.zeros(shape=(len(train_y), output_size))
    new_test_y = np.zeros(shape=(len(test_y), output_size))
    
    for i in range(len(train_y)):
        new_train_y[i][int(train_y[i])] = 1
        
    for i in range(len(test_y)):
        new_test_y[i][int(test_y[i])] = 1
        
    train_y = new_train_y
    test_y = new_test_y
    
	# Training and validation split. (%80-%20)
    valid_x = np.asarray(train_x[int(0.8*len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.8*len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.8*len(train_x))])
    train_y = np.asarray(train_y[0:int(0.8*len(train_y))])
    
    train(train_x, train_y, valid_x, valid_y)
    print("Test Scores:")
    print(test(test_x, test_y,U,W))

    
