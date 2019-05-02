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
from keras.layers import Input, Dense , LSTM , Bidirectional , ConvLSTM2D , concatenate

from keras.callbacks import Callback as Callback



#init encoder for ss
sspath = '/home/cactuskid/Dropbox/machine_learning/SSdataset/'
fastas = glob.glob(sspath +'*.txt')
print(fastas)

import tensorflow as tf
from keras.backend import tensorflow_backend as K

with tf.Session(config=tf.ConfigProto( intra_op_parallelism_threads=4)) as sess:
	K.set_session(sess)

	#window of amino acids to inspect in lstm
	#must be an odd number

	windowlen = 21

	#first part of the network, layered lstm
	LSTMoutdim = 200
	LSTMlayers = 5

	#second part of the network, dense decoder
	Denselayers = 5
	Denseoutdim = 200

	#cutoff for fft input
	fft_freqcutoff = 100

	#save itnerval
	saveinterval = 100

	#times to run the training set through
	epochs = 2

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
	prot2prot2secSingle = functools.partial( SSfunctions.seq2vec , propdict= propdict, verbose =False )

	generator = SSfunctions.datagenerator(fastas , singleSec2vec = prot2prot2secSingle, clipfft =fft_freqcutoff,windowlen= windowlen ,  embeddingprot = prot2sec , embeddingSS = ssencoder , conv2dlstm = conv2dlstm, verbose = False )

	#define model
	#multilayer biderictional stateful lstm

	#todo: change to one categorical input and one matrix inputThe truncated or zero-padded input, transformed along the axes indicated by axes, or the last two axes if axes is not given.

	layeroutputs = {}

	if conv2dlstm == False:
		#(batch_size, timesteps, input_dim)
		window = Input(name='seqin', batch_shape=(1, windowlen, len(propdict)  )  )
	else:
		window = Input(name='seqin', batch_shape=(1,1,windowlen, len(propdict) , 1 ) )

	#input is the size of one sequence windowelse:try:
	#x = concatenate([x,inputs], axis=-1)
	position1 = Input(name='position1' , batch_shape = (1,windowlen, 1) )

	inputs = concatenate([window, position1], axis= -1 )

	for n in range(LSTMlayers):
		if n == 0:
			if conv2dlstm == False:
				layer = LSTM(LSTMoutdim ,name='lstm_'+str(n),activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=True,
				return_state=False, go_backwards=False, stateful=True, unroll=True)
				x = layer(inputs)

				layer = LSTM(LSTMoutdim ,name='lstm_'+str(n)+'back',activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=True,
				return_state=False, go_backwards=True, stateful=True, unroll=True)
				x = layer(x)

				layer = Bidirectional(layer, merge_mode='concat', weights=None)

			else:
				layer = ConvLSTM2D(LSTMoutdim, kernel_size, name ='lstm_'+str(n)+'back', strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
				 bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=True,
				 go_backwards=False, stateful=True, dropout=0.0, recurrent_dropout=0.0)

			x = layer(x)
			#print(inputs)
		else:
			if conv2dlstm == False:
				layer = LSTM(LSTMoutdim, activation='tanh', name='lstm_'+str(n), dropout=0.1, recurrent_dropout=0.1, return_sequences=True, return_state=False, go_backwards=False, stateful=True, unroll=True)
				#layer = Bidirectional(layer, merge_mode='concat', weights=None)
			else:
				layer = ConvLSTM2D(LSTMoutdim, kernel_size, name ='lstm_'+str(n), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
				 bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=True,
			 go_backwards=False, stateful=True, dropout=0.0, recurrent_dropout=0.0)
				x = layer(x)

				layer = ConvLSTM2D(LSTMoutdim, kernel_size, name ='lstm_'+str(n)+'back', strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
				  bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=True,
			  go_backwards=True, stateful=True, dropout=0.0, recurrent_dropout=0.0)
			  	x = layer(x)

				layer = Bidirectional(layer, merge_mode='concat', weights=None)

			x = layer(x)
		layeroutputs[n]= x
	layer = LSTM(LSTMoutdim ,name='lstm_final' ,activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=True, go_backwards=False, stateful=True, unroll=True)
	x,_,_ = layer(x)
	layeroutputs[LSTMlayers+1]= x
	auxiliary_output = Dense(8, activation='softmax', name='aux_output')(x)
	layeroutputs['aux']= auxiliary_output


	#add the fft of whole prot here before the deep layer
	fftin = Input(name='fftin', batch_shape=(1, len(propdict) * fft_freqcutoff) )
	position2 = Input(name='position2' , batch_shape = (1,1) )

	x = concatenate([x,fftin, position2], axis=-1)

	#return state of lstm and fft to dense layers for categorical_crossentropy
	for n in range(Denselayers):
		if n == 0 :
			x = Dense( Denseoutdim , name = 'Dense_'+str(n) , activation='tanh')(x)
			layeroutputs['Dense_'+str(n)] = x
		else:
			x = Dense(Denseoutdim,  name = 'Dense_'+str(n) ,  activation='tanh')(x)
			layeroutputs['Dense_'+str(n)] = x

	#output to SS pred
	layer = Dense(8, name = 'output', activation = 'softmax')
	output = layer(x)

	layeroutputs['final']= x


	#try to concatenate outputs of the batch
	model = Model(inputs = [window,fftin, position1, position2] ,outputs = [auxiliary_output , output])
	model.compile( optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
	max_len = 3000

	#dense output layer with categorical cross entropy error
	traincount = 0
	testcount=0
	print('start taining')
	TRAIN_epochs =1
	train_samples = 100
	testsamples = 10

	cycles = 20

	#set keras session to tf here.

	train_accuracy_rnn=[]
	train_loss_rnn=[]

	train_accuracy_deep=[]
	train_loss_deep=[]



	test_accuracy_rnn=[]
	test_loss_rnn=[]

	test_accuracy_deep=[]
	test_loss_deep=[]


	for k in range(cycles):
		####train
		for i in range(train_samples):
			ID,X,FFTX,Y = next(generator)
			traincount+=1

			#learn in stateful mode over windows

			loss_rnn=[]
			loss_deep=[]
			accuracy_rnn=[]
			accuracy_deep=[]

			for j in range(TRAIN_epochs):
				for i in range(X.shape[0]):
					position = np.asarray([ i - (windowlen -1)/2 + step for step in range(windowlen) ])/(X.shape[0]+(windowlen-1)/2)


					performance = model.train_on_batch( [np.expand_dims(X[i,:,:], axis=0),np.expand_dims( FFTX, axis =0 ) ,position[np.newaxis ,: , np.newaxis] , np.asarray([i/(X.shape[0]+(windowlen-1)/2)] ) ] ,  [ Y[i,:], Y[i,:]] )
					accuracy_rnn.append(performance[3])
					accuracy_deep.append(performance[4])
					loss_rnn.append(performance[1])
					loss_deep.append(performance[2])
			print(ID)

			dl = np.mean(loss_deep)
			ad = np.mean(accuracy_deep)

			lr = np.mean(loss_rnn)
			ar = np.mean(accuracy_rnn)

			train_accuracy_rnn.append(ar)
			train_accuracy_deep.append(ad)
			train_loss_rnn.append(lr)
			train_loss_deep.append(dl)

			print('accuracy training RNN = {}'.format(ar))
			print('loss training RNN = {}'.format(lr))

			print('accuracy training deep = {}'.format(ad))
			print('loss training deep = {}'.format(dl))

			#reset for next sequence
			loss_rnn=[]
			loss_deep=[]
			accuracy_rnn=[]
			accuracy_deep=[]
			model.reset_states()
			###save
		# serialize model to JSON
		model_json = model.to_json()
		with open(str(cycles)+"model.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(str(cycles)+"model.h5")
		print("Saved model to disk")


		#### test
		for j in range(testsamples):
			ID,X,FFTX,Y = next(generator)
			testcount+=1

			#learn in stateful mode over windows
			predy2=[]
			predy1=[]

			for j in range(TRAIN_epochs):
				for i in range(X.shape[0]):
					position = i/X.shape[0]
					if verbose_train == True:
						print( np.expand_dims(X[i,:,:], axis=0).shape)
						print(np.expand_dims(Y[i,:], axis=0).shape )
					y1,y2 = model.predict_on_batch( [np.expand_dims(X[i,:,:], axis=0),np.expand_dims( FFTX, axis =0 ) , position , position] )
					predy1.append(y1)
					predy2.append(y2)
			print(ID)

			print(predy1)
			print(predy2)
			YRNN = np.concatenate(y2)
			YDEEP = np.concatenate(y1)




			print('accuracy training RNN = {}'.format(ar))
			print('loss training RNN = {}'.format(lr))

			print('accuracy training deep = {}'.format(ad))
			print('loss training deep = {}'.format(dl))
			#reset for next sequence
			loss_rnn=[]
			loss_deep=[]
			accuracy_rnn=[]
			accuracy_deep=[]
			model.reset_states()
