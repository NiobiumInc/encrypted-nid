# Copyright (c) 2017 Yusuke Sugomori
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Portions of this code have been adapted from Yusuke Sugomori's code on GitHub: https://github.com/yusugomori/DeepLearning

import numpy

def sigmoid(x):
        return 1./ (1 + numpy.exp(-x))

class dA_logistic:
    def __init__(self):
        #fit Chebyshev approximations to sigmoid and tanh
        #create input array
        x = numpy.ones(1200)
        for ii in range(0, 1200):
            x[ii] = float(ii - 600)/100.0

        # find sigmoid and tanh over input
        y = sigmoid(x)
        z = numpy.tanh(x)

        self.order = 5

        #fit +/- 5 range for sigmoid, order 5
        self.sig_apx = numpy.polynomial.chebyshev.chebfit(x[100:1100],y[100:1100], self.order)

        #fit +/- 2 range for tanh, order 5
        self.tanh_apx = numpy.polynomial.chebyshev.chebfit(x[400:800],z[400:800], self.order)

    def get_sigmoid_apx(self):
        return self.sig_apx
    
    def get_tanh_apx(self):
        return self.tanh_apx
    
    def get_order(self):
        return self.order


class dA_params:
    def __init__(self, n_visible = 5, n_hidden = 3, lr=0.001, hiddenRatio=None, chebyapx=None, nonlin='sigmoid'):
        self.n_visible = n_visible# num of units in visible (input) layer
        self.lr = lr
        self.hiddenRatio = hiddenRatio
        if hiddenRatio is not None:
            self.n_hidden = int(numpy.ceil(self.n_visible*self.hiddenRatio))
        else:
            self.n_hidden = int(n_hidden)  # num of units in hidden layer

        self.chebypoly = chebyapx
        self.nonlin = nonlin

class dA:

    def __init__(self, params):
        self.params = params
 
        self.rng = numpy.random.RandomState(1234)

        a = 1. / self.params.n_visible
        self.W = numpy.array(self.rng.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(self.params.n_visible, self.params.n_hidden)))

        self.hbias = numpy.zeros(self.params.n_hidden)  # initialize hidden layer bias 0
        self.rbias = numpy.zeros(self.params.n_visible)  # initialize reconstruction layer bias 0
        self.W_prime = self.W.T

        # instrument AEs to find extent of values into the sigmoid
        self.innermax = -1000.0*numpy.ones(self.params.n_hidden)
        self.innermin = 1000.0*numpy.ones(self.params.n_hidden)
        self.outermax = -1000.0*numpy.ones(self.params.n_visible)
        self.outermin = 1000.0*numpy.ones(self.params.n_visible)


     # Encode
    def get_hidden_values(self, input):
        siginput = numpy.dot(input, self.W) + self.hbias

        newmax = siginput > self.innermax
        self.innermax[newmax] = siginput[newmax]
        newmin = siginput < self.innermin
        self.innermin[newmin] = siginput[newmin]

        return siginput

    # Decode
    def get_reconstructed_input(self, hidden):
        siginput = numpy.dot(hidden, self.W.T) + self.rbias

        newmax = siginput > self.outermax
        self.outermax[newmax] = siginput[newmax]
        newmin = siginput < self.outermin
        self.outermin[newmin] = siginput[newmin]

        return siginput
 
    def train(self, x):
        #self.n = self.n + 1

        yraw = self.get_hidden_values(x)
        if self.params.nonlin=='tanh':
           y = numpy.tanh(yraw)
        else:
            y = sigmoid(yraw)
        
        zraw = self.get_reconstructed_input(y)
        if self.params.nonlin=='tanh':
           z = numpy.tanh(zraw)
        else:
            z = sigmoid(zraw)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)

        L_rbias = L_h2
        L_hbias = L_h1
        L_W = numpy.outer(x.T, L_h1) + numpy.outer(L_h2.T, y)

        self.W += self.params.lr * L_W
        self.hbias += self.params.lr * L_hbias
        self.rbias += self.params.lr * L_rbias
        return L_h2 #the raw squared reconstruction error vector during training
 
    
    def execute(self, x): #returns MSE of the reconstruction of x
        yraw = self.get_hidden_values(x)
        if self.params.chebypoly is None:
            if self.params.nonlin=='tanh':
                y = numpy.tanh(yraw)
            else:
                y = sigmoid(yraw)
        else:
            y = numpy.polynomial.chebyshev.chebval(yraw, self.params.chebypoly)

        zraw = self.get_reconstructed_input(y)
        if self.params.chebypoly is None:
            if self.params.nonlin=='tanh':
                z = numpy.tanh(zraw)
            else:
                z = sigmoid(zraw)
        else:
            z = numpy.polynomial.chebyshev.chebval(zraw, self.params.chebypoly)

        rawdif = x-z
        return rawdif       #raw reconstruction error vector during inference
    
    def get_limits(self):
        return (self.innermax, self.innermin, self.outermax, self.outermin)
    
    def get_parameters(self):
        return(self.W, self.hbias, self.rbias)
    
    def load_parameters(self, newW, newhbias, newrbias):
        self.W = newW
        self.hbias = newhbias
        self.rbias = newrbias
    
   