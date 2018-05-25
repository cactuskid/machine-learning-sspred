import pandas as pd
import numpy as np
import keras
from Bio import SeqIO
import glob
import functools
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Input, Dense , LSTM , Bidirectional

def protsec2numpy(sec, windowlen,verbose= False):

	#window should be an odd number
	#slice up the sequence into window sized chunks
	padding = (windowlen-1)/2
	originallen = len(sec)
	sec = ['-']*int(padding) + list(sec) + int(padding)*['-']
	sechunks=[ np.asarray(str(sec[i:i + windowlen]).encode()) for i in range(0, originallen)]
	if verbose == True:
		print(sechunks)
	return sechunks


def econdedDSSP(ssStr,intdico,encoder):
	'''
	H = alpha-helix
	B = residue in isolated beta-bridge
	E = extended strand, participates in beta ladder
	G = 3-helix (310 helix)
	I = 5 helix (pi-helix)
	T = hydrogen bonded turn
	S = bend
	'''
	try:
		intvec = np.asarray([ intdico[char] for char in ssStr])

		onehot = encoder.transform(intvec.reshape(-1, 1))

		return onehot

	except:
		print(ssStr)
		return(ssStr)
def datagenerator(fastas , n=100 , windowlen= 13, embeddingprot=None , embeddingSS=None):
	#yield string for x and y to make a df block of n sequences to learn with
	for fasta in fastas:
		fastaIter = SeqIO.parse(fasta, "fasta")
		seqDict={}
		for seq in fastaIter:
			chainID = str(seq.description)
			ID = chainID[0:6]
			if ID not in seqDict:
				seqDict[ID]= {}
			if 'secstr' in seq.description:
				seqDict[ID]['SS']= str(seq.seq)
			else:
				seqDict[ID]['AA']= str(seq.seq)

			if len(seqDict)>n:
				df = pd.DataFrame.from_dict(seqDict, orient= 'index')
				df['X'] = df['AA'].map(embeddingprot)
				df['Y'] = df['SS'].map(embeddingSS)
				yield df[ df.Y.notna()]
				seqDict={}

#init encoder for ss
sspath = './SSdataset/'
fastas = glob.glob(sspath + '*.txt')
print(fastas)

windowlen = 13

LSTMoutdim = 30
LSTMlayers = 3

Denselayers = 3
Denseoutdim = 30

saveinterval = 100


encoder = OneHotEncoder()
encoder.fit( np.asarray( np.asarray(range(7)).reshape(-1,1) ) )

intdico = { charval : int(i) for i,charval in enumerate(['H', 'B', 'E', 'G', 'I', 'T', 'S'] ) }
ssencoder = functools.partial( econdedDSSP , intdico= intdico , encoder = encoder)
prot2sec = functools.partial( protsec2numpy , windowlen= windowlen )

generator = datagenerator(fastas, n = 10 , windowlen= windowlen ,  embeddingprot = prot2sec , embeddingSS = ssencoder)
print(next(generator))

#define model
#multilayer biderictional stateful lstm
layeroutputs = {}
#input is the size of one sequence window
for n in range(LSTMlayers):
	if n == 0:
		layer = LSTM(outdim,input_dim=1, input_length=windowlen,  name='lstm_'+str(n), activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=True, return_state=False, go_backwards=False, stateful=True, unroll=False)
		layer = Bidirectional(layer, merge_mode='concat', weights=None)
		x = layer(inputs)
	else:
		layer = LSTM(outdim, activation='tanh', name='lstm_'+str(n), dropout=0.0, recurrent_dropout=0.0, return_sequences=True, return_state=False, go_backwards=False, stateful=True, unroll=False)
		layer = Bidirectional(layer, merge_mode='concat', weights=None)
		x = layer(x)
	layeroutputs[n]= x
layer = LSTM(outdim, activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=True, unroll=False)
layer = Bidirectional(layer, merge_mode='concat', weights=None)

x = layer(x)
layeroutputs[LSTMlayers+1]= x
auxiliary_output = Dense(7, activation='softmax', name='aux_output')(x)
layeroutputs['aux']= auxiliary_output

#return state of lstm to dense layers
for n in range(Denselayers):
	if n == 0 :
		x = Dense( Denseoutdim , name = 'Dense_'+str(n))(x)
		layeroutputs['Dense_'+str(n)] = x
	else:
		x = Dense(Denseoutdim,  name = 'Dense_'+str(n))(x)
		layeroutputs['Dense_'+str(n)] = x

#output to SS pred
layer = Dense(7, name = 'output', activation = 'softmax')
output = layer(x)
layeroutputs['final']= x
model = Model(inputs = inputs , outputs = [auxiliary_output, output])

#decode topology
model.compile( optimizer='RMSprop', loss='categorical_cross_entropy', metrics=['acc'])

#dense output layer with categorical cross entropy error
for i,seqDF in enumerate(generator):
	for row in seqDF.items():
		#learn in stateful mode over windows
		model.train_on_batch(row.X , row.Y, verbose = True)
		#reset
		model.reset_states()
	if i % saveinterval == 0:
		model.save()
