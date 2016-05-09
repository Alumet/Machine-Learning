# @Author: Yann Huet
# @Date:   2015-07-11T15:19:32+02:00
# @Email:  https://github.com/Alumet
# @Last modified by:   Yann Huet
# @Last modified time: 2016-05-09T17:48:51+02:00
# @License: MIT License (MIT), Copyright (c) Yann Huet

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plot

class Neural_Network(object):
    '''
        Simple Neural Network generator based on NumPy + SciPy
        inspired by Neural Networks Demystified series by "Welch Labs" https://www.youtube.com/watch?v=bxe2T-V8XRs

        - sigmoide activation function
        - stochastic gradient descent
        - back propagation

        ##initialisation:
        NN=Neural_NetWork(ImputLayerSize,HiddenLayerSize,HiddenLayerNumber)

    '''
    def __init__(self,ImputLayerSize=1,HiddenLayerSize=1,HiddenLayerNumber=1):
        #self deffinition of hyperparamaeters
        self.ImputLayerSize=ImputLayerSize
        self.OutputLayerSize=1
        self.HiddenLayerSize=HiddenLayerSize
        self.HiddenLayerNumber=HiddenLayerNumber

        #wheigts (random starting parameters)
        self.wheigts=[]

        weigth_temp=np.random.randn(self.ImputLayerSize, self.HiddenLayerSize)
        self.wheigts.append(weigth_temp)

        for i in range (1,HiddenLayerNumber):
            weigth_temp=np.random.randn(self.HiddenLayerSize, self.HiddenLayerSize)
            self.wheigts.append(weigth_temp)

        weigth_temp=np.random.randn(self.HiddenLayerSize, self.OutputLayerSize)
        self.wheigts.append(weigth_temp)

    #forward processing trougth layers
    def forward(self,A_temp):
        self.Z=[]
        self.A=[]

        for i in range (0,len(self.wheigts)):
            Z_temp=np.array(np.matrix(np.dot(A_temp,self.wheigts[i])))
            A_temp=np.array(np.matrix(self.activation(Z_temp)))

            self.A.append(A_temp)
            self.Z.append(Z_temp)

        return A_temp

    #Sigmoide activation function
    def activation(self,Z):
        return 1/(1+np.exp(-Z))

    #Prime activation function for back propagation
    def activationPrime(self,Z):
        return np.exp(-Z)/((1+np.exp(-Z))**2)

    def costFunction(self,X,Y):
        self.results=self.forward(X)
        sub_temp=np.subtract(Y,self.results)
        cost=0.5*sum(sub_temp**2)
        cost=cost/X.shape[0]+(self.penalty/2)*(sum(self.getParams()**2))

        return cost

    def costFunctionPrime(self,X,Y):

        X=np.array(np.matrix(X))
        self.results=self.forward(X)
        sub_temp=-(np.subtract(Y,self.results))
        sub_temp=np.array(np.matrix(sub_temp))
        dJdW=[]

        delta_0=np.multiply(sub_temp,self.activationPrime(self.Z[len(self.Z)-1]))
        dJdW_temp=np.dot(self.A[len(self.A)-2].T,delta_0)
        dJdW_temp=dJdW_temp/X.shape[0]+self.penalty*self.wheigts[len(self.wheigts)-1]
        dJdW.append(dJdW_temp)

        for i in range (self.HiddenLayerNumber-1,0,-1):
            delta_0=np.dot(delta_0,self.wheigts[i+1].T)*self.activationPrime(self.Z[i])
            dJdW_temp=np.dot(self.A[i-1].T,delta_0)
            dJdW_temp=dJdW_temp/X.shape[0]+self.penalty*self.wheigts[i+1]
            dJdW.append(dJdW_temp)

        delta_0=np.dot(delta_0,self.wheigts[1].T)*self.activationPrime(self.Z[0])
        dJdW_temp=np.dot(X.T,delta_0)
        dJdW_temp=dJdW_temp/X.shape[0]+self.penalty*self.wheigts[0]
        dJdW.append(dJdW_temp)

        dJdW=dJdW[::-1]
        return dJdW

    #Get and format Neural_NetWork parameters for SciPy
    def getParams(self):
        params=np.array([])
        for wheigt in self.wheigts:
            params=np.concatenate((params,wheigt.ravel()))
        return params

    #Get and format Neural_NetWork for SciPy
    def getGrad(self,X,Y):
        dJdW=self.costFunctionPrime(X,Y)
        grad=np.array([])
        for djdw in dJdW:
            grad=np.concatenate((grad,djdw.ravel()))
        return grad

    #Return cost
    def getCost(self,X,Y):
        J=sum(self.costFunction(X,Y))
        return J

    #Set parmeters after SciPy optimization
    def setParams(self,params):
        cuts=[[0,self.HiddenLayerSize*self.ImputLayerSize,self.ImputLayerSize,self.HiddenLayerSize]]

        for i in range (0,self.HiddenLayerNumber-1):
            cuts.append([cuts[i][1],cuts[i][1]+self.HiddenLayerSize**2,self.HiddenLayerSize,self.HiddenLayerSize])

        cuts.append([cuts[len(cuts)-1][1],cuts[len(cuts)-1][1]+self.HiddenLayerSize*self.OutputLayerSize,self.HiddenLayerSize,self.OutputLayerSize])

        wheigts_temp=[]
        for cut in cuts:
            wheigts_temp.append(np.reshape(params[cut[0]:cut[1]],(cut[2],cut[3])))
        self.wheigts=wheigts_temp

    #Wrap cost function SciPy
    def costFunctionWraper(self,params,X,Y):

        self.setParams(params)
        cost=self.getCost(X,Y)
        grad=self.getGrad(X,Y)


        return cost, grad

    #
    def callback(self,params):
        self.setParams(params)
        self.train_cost.append(self.costFunction(self.X,self.Y))

        if "testing" in self.options:
            self.train_cost_test.append(self.costFunction(self.test_X,self.test_Y))

    #train Neural_NetWork on training set
    '''
        Neural_NetWork.train(data,penalty)

        Penalty is needed only if Data_amout > (HiddenLayerSize * HiddenLayerNumber)
    '''
    def train(self,data,penalty=0.0001):

        X,X_test,Y,Y_test=data
        self.X=X
        self.test_X=X_test
        self.Y=Y
        self.test_Y=Y_test
        self.penalty=penalty

        self.train_cost=[]
        self.train_cost_test=[]

        params0=self.getParams()

        #SciPy optimization
        options={'maxiter':200,'disp':True}
        _res=optimize.minimize(self.costFunctionWraper,params0, jac=True, method='BFGS',args=(X,Y),options=options,callback=self.callback)

        self.setParams(_res.x)
        self.optimization=_res

        plot.plot(NN.train_cost)

        if "testing" in self.options:
            plot.plot(NN.train_cost_test)
        plot.grid(1)
        plot.xlabel('Iterations')
        plot.ylabel('cost')

    #data preparation
    '''
    Neural_NetWork.prepData(X training data,Y training data,percentage=0.8,options=())

    options:
    - "scuffle" : schuffle randomly training data
    - "testing" : creat data subset to test Neural_NetWork perfs (percentage variable to cut data detfault is 80% training, 20% testing)
                  testing helps to detect overfiting
    - "normalize" : Normalize dataset betxeen 0 and 1 for better perfs

    '''
    def prepData(self,X,Y,percentage=0.8,options=()):

        self.options=options

        _test_X=[]
        _test_Y=[]

        if "schuffle" in options:
            T=np.concatenate((X,Y),axis=1)
            np.random.shuffle(T)


            if self.ImputLayerSize>1:
                X=T[:,(0,self.ImputLayerSize-1)]
            else:
                X=np.array(np.matrix(T[:,0])).T

            if self.OutputLayerSize>1:
                Y=T[:,(self.ImputLayerSize,self.ImputLayerSize+self.OutputLayerSize-1)]
            else:
                Y=np.array(np.matrix(T[:,(self.ImputLayerSize)])).T

        if "testing" in options:
            _X=np.reshape(X[0:int(len(X)*percentage)],(int(len(X)*percentage),self.ImputLayerSize))
            _test_X=np.reshape(X[int(len(X)*percentage):len(X)],(len(X)-int((len(X)*percentage)),self.ImputLayerSize))
            _Y=np.reshape(Y[0:int(len(Y)*percentage)],(int(len(Y)*percentage),self.OutputLayerSize))
            _test_Y=np.reshape(Y[int(len(Y)*percentage):len(Y)],(len(Y)-int((len(Y)*percentage)),self.OutputLayerSize))
        else:
            _X=X
            _Y=Y

        if "normalize" in options:
            _X=_X/np.amax(_X,axis=0)
            _Y=_Y/100

            if "testing" in options:
                _test_X=_test_X/np.amax(_test_X,axis=0)
                _test_Y=_test_Y/100

        return _X,_test_X,_Y,_test_Y


#NN creation
NN=Neural_Network(2,3,1)

#data preparation
X=np.array(([3,5],[5,1],[10,2],[6,1.5],[4,5.5],[4.5,1],[9,2.5],[6,2]),dtype=float)
Y=np.array(([75],[82],[93],[70],[70],[89],[85],[75]),dtype=float)

data=NN.prepData(X,Y,0.8,("testing","normalize","schuffle"))

#training
NN.train(data,1e-4)


print(np.subtract(NN.forward(data[0]),data[2]))
print(np.subtract(NN.forward(data[1]),data[3]))

plot.show()
