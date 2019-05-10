import pandas as pd
import numpy as np
import keras
from Bio import SeqIO
import glob
import pdb
import functools
import functions
import itertools

from scipy import signal as sig
from scipy.stats import bernoulli
from scipy.signal import gaussian
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Input, Dense , LSTM , Bidirectional

def protsec2numpy(sec, windowlen, propdict=None, verbose= False):
	#window should be an odd number
	#slice up the sequence into window sized chunks
	padding = (windowlen-1)/2
	try:
		originallen = len(sec)
	except TypeError:
		return None
	sec = ''.join(['-']*padding) + sec + ''.join(padding*['-'])
	if verbose == True:
		print(sec)
		print(len(sec))
		print(type(sec))
	sechunks=  [ [ x for x in str(sec[i:i + windowlen])]  for i in range(0, originallen)]
	if verbose == True:
		pass
		#print(sechunks)
	#transform to discrete vals of physical physicalProps_pipeline
	if propdict == None:
		return sechunks
	else:
		matrices = [ seq2vec(chunk, propdict=propdict, verbose= verbose) for chunk in sechunks]
		if verbose == True:
			print(propdict)
			print(matrices[0])
		return matrices

def seq2vec(seq, propdict, verbose = False):
	#countmat is length of sequence
	if verbose == True:
		print(seq)
		print(type(seq))
		print(len(seq))
	propmat = np.zeros(( len(propdict) , len(seq) ))

	for i,prop in enumerate(propdict):
		vals = [ propdict[prop][char] if char in propdict[prop] else 0 for char in seq ]
		propmat[i,:] = vals


	if verbose == True:
		print(propmat.T)

	return propmat.T

def econdedDSSP(ssStr,intdico,encoder, verbose= False):

	'''

	H = alpha-helix
	B = residue in isolated beta-bridge
	E = extended strand, participates in beta ladder
	G = 3-helix (310 helix)
	I = 5 helix (pi-helix)
	T = hydrogen bonded turn
	S = bend
	' ' = nopred

	'''

	try:
		if verbose == True:
			print(str(ssStr))

		intvec = np.asarray([ intdico[char] for char in ssStr])
		onehot = encoder.transform(intvec.reshape(-1, 1))

		return onehot
	except:
		print('err')
		print(ssStr)
		return ssStr

def fastaparse(fastafile):
	sequence = {}
	with open(fastafile, 'r') as file:
		for line in file:
			if '>' in line:
				if len(sequence)> 0:
					yield sequence
					sequence = {}
 				sequence['description']= line[1:]
			else:
				if 'seq' not in sequence:
					sequence['seq'] = line.replace('\n', '')
				else:
					sequence['seq']+= line.replace('\n', '')

def datagenerator(fastas , windowlen= 13, clipfft = 500 , p = .01 ,singleSec2vec = None, embeddingprot=None , embeddingSS=None , Testmode=False, conv2dlstm = False, verbose = False):
	#yield string for x and y to make a df block of n sequences to learn with
	nGaussian = 7
	stdv = .05
	Gaussian = gaussian(nGaussian, stdv)

	for fasta in itertools.cycle(fastas):
		fastaIter = fastaparse(fasta)
		seqDict={}
		for seq in fastaIter:
			#select random entries
			r = bernoulli.rvs(p, size=1 )

			if sum(r) == 1:
				chainID = seq['description']
				ID = chainID[0:6]

				if ID not in seqDict:
					seqDict[ID]= {}

				if 'secstr' in seq['description']:
					if verbose == True:
						print(len(seq['seq']))
					seqDict[ID]['SS']= str(seq['seq'])

				else:
					seqDict[ID]['AA']= str(seq['seq'])
					if verbose == True:
						print(len(seq['seq']))

				if len(seqDict[ID])>1:
					X = embeddingprot(seqDict[ID]['AA'])
					Y = embeddingSS(seqDict[ID]['SS']).todense()
					X = np.stack(X, axis =0)
					#fft of 2d feature matrix of full sequence
					Xfull = singleSec2vec(seqDict[ID]['AA'])


					for i in range(Xfull.shape[0]):
						Xfull[i,:] = sig.fftconvolve(Xfull[i,:], Gaussian , mode='same')

					FFTX = np.fft.fft2(Xfull).T
					print(FFTX.shape)
					if FFTX.shape[1]-1 < clipfft:
						FFTX = np.hstack( [FFTX , np.zeros( ( FFTX.shape[0] , clipfft - FFTX.shape[1] ))] )

					FFTX= FFTX[:, 0:clipfft].ravel()

					#add a fe descriptors of the overall data here
					#%each amino acid
					#PI
					#
					count = [len(X), ]

					if verbose == True:
						print(FFTX.shape)
						print(Y.shape)
						print(X.shape)

					if X.shape[0] > 1:
							yield ID,X,FFTX, Y
					seqDict={}
