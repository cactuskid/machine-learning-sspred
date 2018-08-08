import SSfunctions
import functions

import pandas as pd
import numpy as np
import keras
from Bio import SeqIO
import glob
import pdb
import functools
import functions

from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Input, Dense , LSTM , Bidirectional , ConvLSTM2D

from keras.callbacks import Callback as Callback

##  TODO: balance dataset with clustering
##  TODO: implement convolution LSTM
##  TODO: output to mat or input in next network
##  TODO: 



#init encoder for ss
sspath = '/home/cactuskid/Dropbox/machine_learning/SSdataset/'
fastas = glob.glob(sspath +'*.txt')
print(fastas)

#window of amino acids to inspect in lstm
#must be an odd number

windowlen = 21

#first part of the network, layered lstm
LSTMoutdim = 200
LSTMlayers = 5

#second part of the network, dense decoder
Denselayers = 5
Denseoutdim = 100
#save itnerval
saveinterval = 100

#times to run the training set through
epochs = 1

verbose = False
verbose_train = False


#todo finish conv lstm implementation
conv2dlstm = False
kernel_size = 4



proppath = '/home/cactuskid/Dropbox/machine_learning/physicalpropTable.csv'
propdict = functions.loadDict(proppath)
encoder = OneHotEncoder()

encoder.fit( np.asarray( np.asarray(range(8)).reshape(-1,1) ) )
intdico = { charval : int(i) for i,charval in enumerate(['H', 'B', 'E', 'G', 'I', 'T', 'S' , ' '] ) }
ssencoder = functools.partial( SSfunctions.econdedDSSP , intdico= intdico , encoder = encoder, verbose = False)
prot2sec = functools.partial( SSfunctions.protsec2numpy , windowlen= windowlen , propdict= propdict, verbose =False )
generator = SSfunctions.datagenerator(fastas , windowlen= windowlen ,  embeddingprot = prot2sec , embeddingSS = ssencoder , conv2dlstm = conv2dlstm, verbose = True )

#define model
#multilayer biderictional stateful lstm
layeroutputs = {}

True
#todo: change to one categorical input and one matrix input
if conv2dlstm == False:

	#(batch_size, timesteps, input_dim)
	inputs = Input(name='seqin', batch_shape=(1, windowlen, len(propdict)  )  )
else:

	inputs = Input(name='seqin', batch_shape=(1,1,windowlen, len(propdict) , 1 ) )

#input is the size of one sequence windowelse:try:

for n in range(LSTMlayers):
	if n == 0:
		if conv2dlstm == False:
			layer = LSTM(LSTMoutdim ,name='lstm_'+str(n),activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=True,
			return_state=False, go_backwards=False, stateful=True, unroll=True)
			layer = Bidirectional(layer, merge_mode='concat', weights=None)

		else:
			layer = ConvLSTM2D(LSTMoutdim, kernel_size, name ='lstm_'+str(n), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
			 bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=True,
			 go_backwards=False, stateful=True, dropout=0.0, recurrent_dropout=0.0)

		x = layer(inputs)
		#print(inputs)
	else:
		if conv2dlstm == False:
			layer = LSTM(LSTMoutdim, activation='tanh', name='lstm_'+str(n), dropout=0.0, recurrent_dropout=0.0, return_sequences=True, return_state=False, go_backwards=False, stateful=True, unroll=True)
			layer = Bidirectional(layer, merge_mode='concat', weights=None)

		else:
			layer = ConvLSTM2D(LSTMoutdim, kernel_size, name ='lstm_'+str(n), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
			 bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=True,
			 go_backwards=False, stateful=True, dropout=0.0, recurrent_dropout=0.0)

		x = layer(x)
	layeroutputs[n]= x
layer = LSTM(LSTMoutdim ,name='lstm_final' ,activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=True, unroll=True)

#layer = Bidirectional(layer, merge_mode='concat', weights=None)
x = layer(x)
layeroutputs[LSTMlayers+1]= x
auxiliary_output = Dense(8, activation='softmax', name='aux_output')(x)
layeroutputs['aux']= auxiliary_output

#return state of lstm to dense layerscategorical_crossentropy
for n in range(Denselayers):
	if n == 0 :
		x = Dense( Denseoutdim , name = 'Dense_'+str(n))(x)
		layeroutputs['Dense_'+str(n)] = x
	else:
		x = Dense(Denseoutdim,  name = 'Dense_'+str(n))(x)
		layeroutputs['Dense_'+str(n)] = x
#output to SS pred
layer = Dense(8, name = 'output', activation = 'softmax')
output = layer(x)

layeroutputs['final']= x
#decode topology

model = Model(inputs = inputs , outputs = [auxiliary_output , output])
model.compile( optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
max_len = 3000


#dense output layer with categorical cross entropy error
traincount = 0
print('start taining')

#model.fit_generator( generator=generator , callbacks=[ResetStatesCallback()], steps_per_epoch=1 , epochs= 10000 , verbose = 2 , shuffle=False)


for i,mats in enumerate(generator):
	ID,X,Y = mats
	traincount+=1

	#learn in stateful mode over windows
	#model.fit( X , [SS,SS], callbacks=[ResetStatesCallback()], batch_size=1, shuffle=False, epochs=3)
	accuracy_rnn=[]
	loss_rnn=[]

	accuracy_deep=[]
	loss_deep=[]
	for j in range(epochs):
		for i in range(X.shape[0]):
			if verbose_train == True:
				print( np.expand_dims(X[i,:,:], axis=0).shape)
				print(np.expand_dims(Y[i,:], axis=0).shape )
			performance = model.train_on_batch( np.expand_dims(X[i,:,:], axis=0) , [ Y[i,:], Y[i,:]] )
			accuracy_rnn.append(performance[3])
			accuracy_deep.append(performance[4])
			loss_rnn.append(performance[1])
			loss_deep.append(performance[2])
	print(ID)

	print('accuracy training RNN = {}'.format(np.mean(accuracy_rnn)))
	print('loss training RNN = {}'.format(np.mean(loss_rnn)))

	print('accuracy training deep = {}'.format(np.mean(accuracy_deep)))
	print('loss training deep = {}'.format(np.mean(loss_deep)))
	#reset for next sequence
	loss_rnn=[]
	loss_deep=[]
	accuracy_rnn=[]
	accuracy_deep=[]
	model.reset_states()


	#if i % saveinterval == 0:
	#	model.save()
	#if traincount%est_interval == 0:
		#break and test on validation data here
	#	pass
