# Development of a neural network with two layers for classifying the MNIST Data set
import numpy as np
import gzip
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit

class NeuralNetwork():   
    
    def __init__(self,lr):
        
        #set learning rate
        self.lr=lr
        #initialize weights for the 1st layer 
        self.w1=np.random.randn(785,100)
        #initialize weights for the 2nd layer 
        self.w2=np.random.randn(100,10)

    #cost function
    def cost(self,X_train,y_train):
        #sum of squared erros
        pred=self.predict(X_train)
        v=(1/2)*(pred-y_train)**2/len(y_train)
        c=np.sum(v,axis=0)
        return np.mean(c)

    #activation function
    def activation(self,input):
        #sigmoid function
        return expit(input)
    
    #update weights
    def update(self,X,y):

        #output of hidden layer 
        o1=self.activation(X@self.w1)
        #output of ouput layer 
        o2=self.activation(o1@self.w2) 
        #error of outputlayer   
        error_o2=y-o2
        #propagating error back to hidden layer
        error_o1=error_o2@self.w2.T
        #updating the weighting matrices for hidden and output layer
        dw2=o1.T@(error_o2*o2*(1-o2))/len(X)
        dw1=X.T@(error_o1*o1*(1-o1))/len(X)
        self.w1=self.w1+dw1*self.lr   
        self.w2=self.w2+dw2*self.lr 

    def train(self,X_train,y_train,X_test,y_test,epochs,batchsize):
        scores_train=[]
        scores_test=[]
        losses=[]
        for i in range(0,epochs):
            for j in range(0,60000,batchsize):
                self.update(X_train[j:(j+batchsize),:],y_train[j:(j+batchsize)]) 

            score_train=self.evaluate(X_train,y_train)
            scores_train.append(score_train)
            score_test=self.evaluate(X_test,y_test)           
            scores_test.append(score_test)

            loss=self.cost(X_train,y_train)
            losses.append(loss)

            print("Epoch:                    "+str(i))
            print("Loss:                     "+str(round(loss,4)))
            print("Accuracy on Trainingset:  "+str(round(score_train,4)))
            print(" ")

        plt.figure()
        plt.plot(scores_train)
        plt.plot(scores_test)
        plt.title("Modelscore")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend(["Train","Test"])
        
        plt.figure()
        plt.plot(losses,color="red")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")        
        
    #predict values
    def predict(self,X):

        o1=self.activation(X@self.w1)
        o2=self.activation(o1@self.w2)
        return o2
    
    #model evaluation
    def evaluate(self,X,y):

        pred=self.predict(X)
        pred=np.argmax(pred,axis=1)
        sol=np.argmax(y,axis=1)
        score=np.mean(pred==sol)

        return score

def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16)\
            .reshape(-1, 28, 28)\
            .astype(np.float32)

def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)

oh=OneHotEncoder(categories="auto")
#Open MNIST trainingset and OneHotEncode Output data
X_train = open_images("./mnist/train-images-idx3-ubyte.gz").reshape(-1, 784)/255
y_train = open_labels("./mnist/train-labels-idx1-ubyte.gz").reshape(-1,1)
y_train=oh.fit_transform(y_train.reshape(-1,1)).toarray()
#Add a column with bias to input data
bias=np.ones((60000,1))
X_train=np.append(X_train,bias,axis=1)


#Open MNIST testset and OneHotEncode Outputdata
X_test = open_images("./mnist/t10k-images-idx3-ubyte.gz").reshape(-1, 784)/255
y_test = open_labels("./mnist/t10k-labels-idx1-ubyte.gz")
y_test=oh.transform(y_test.reshape(-1,1)).toarray()
#Add a column with ones to input data
bias=np.ones((10000,1))
X_test=np.append(X_test,bias,axis=1)

#Create a NeuralNetwork and train with data for given number of epochs and certain batchsize
#785 input neurons, 100 neurons in hidden layer, 10 output neurons
model=NeuralNetwork(lr=10)
model.train(X_train,y_train,X_test,y_test,epochs=10,batchsize=500)

plt.show()
