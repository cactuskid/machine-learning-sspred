import pandas as pd
import numpy as np
import keras
from Bio import SeqIO
import glob

from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Input, Dense , LSTM , Bidirectional

def protsec2numpy(sec, windowlen):
	#window should be an odd number
	#slice up the sequence into window sized chunks
	padding = (seclen-1)/2
	originallen = len(sec)
	sec = ['-']*padding + sec + padding*['-']
	sechunks=[ np.asarray(sec[i:i + window].encode()) for i in range(0, originallen)]
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
	intvec = np.assarray([ intdico[char] for char in ssStr])
	onehot = encoder.transform(intvec)
	return onehot

def datagenerator(fastas , n=100 , windowlen= 11, embeddingprot=None , embeddingSS=None):
	#yield string for x and y to make a df block of n sequences to learn with
	for fasta in fastas:
		fastaIter = SeqIO.parse(fasta, "fasta")
		for seq in fastaIter:
			description = seq.description
			ID = chainID[0:6]
			if ID not in seqDict:
				seqDict[ID]= {}
			if 'secstr' in description:
				seqDict[ID]['SS']= str(seq.seq)
			else:
				seqDict[ID]['AA']= str(seq.seq)
			if len(seqDict)>n:
					df = pd.from_dict(seqDict, orient= 'columns')
					df['X'] = df['AA'].map(embeddingprot)
					df['Y'] = df['SS'].map(embeddingSS)
					yield df
					seqDict={}

#init encoder for ss
sspath = './SSdataset/'
fastas = glob.glob(sspath + '*.fasta')
print(fastas)

windowlen = 11
outdim = 30
layers = 3
saveinterval = 100


encoder = OneHotEncoder()
encoder.fit( np.asarray(range(7)) )
intdico = { charval : i for i,charval in enumerate(['H', 'B', 'E', 'G', 'I', 'T', 'S'] ) }
ssencoder = functools.partial( econdedDSSP , intdico= intdico , encoder = encoder)
prot2sec = functools.partial( protsec2numpy , windowlen= windowlen )

generator = datagenerator(fastas, n = 10 , windowlen= windowlen ,  embeddingprot = embeddingprot , embeddingSS = ssencoder)
print(next(generator))
#define model
#multilayer biderictional stateful lstm
model = Model()
layers = []
#input is the size of one sequence window
layer = Input((windowsize,))
layers.append(layer)
for n in range(nlayers):
	layer = LSTM(outdim, activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=True, return_state=False, go_backwards=False, stateful=True, unroll=False)
	layer = Bidirectional(layer, merge_mode='concat', weights=None)
	layers.append(layer)
layer = LSTM(outdim, activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=True, unroll=False)
layer = Bidirectional(layer, merge_mode='concat', weights=None)
#return state of lstm to dense layers
Dense(7,activation = 'softmax')
#decode topology

for layer in layers:
	model.add(layer)
else:
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
