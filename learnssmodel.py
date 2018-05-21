import pandas as pd
import numpy as np
import keras

from Bio import SeqIO



def fastasToDF(fastas , verbose=False):
	regex = re.compile('[^a-zA-Z0-9]')
	regexAA = re.compile('[^ARDNCEQGHILKMFPSTWYV]')
	DFdict={}
	count = 0
	total = []
	DDF =None
	for fasta in fastas:
		fastaIter = SeqIO.parse(fasta, "fasta")
		for seq in fastaIter:
		#	seqstr = regexAA.sub('', str(seq.seq))
		#	desc =str(seq.description)
			fastastr = '>'+str(seq.desc)+'\n'+str(seq.seq)+'\n'
			if desc not in total:
				#check for duclipcates within a folder
				total.append(desc)
				DFdict[desc] = { 'desc': desc.encode(), 'seq':seqstr, 'fasta': fastastr}
			count +=1
			if count % 400 == 0:
				df = pd.DataFrame.from_dict(DFdict, orient = 'index' )
				if df is not None and len(df)>0:
					if DDF is None:
						DDF = dd.from_pandas(df , chunksize = 200)
					else:
						DDF = dd.concat([ DDF,  dd.from_pandas(df , chunksize = 200) ] , interleave_partitions=True )
				DFdict={}
		else:
			df = pd.DataFrame.from_dict(DFdict, orient = 'index')
			if df is not None and len(df)>0:
				if DDF is None:
					DDF = dd.from_pandas(df , chunksize = 200)
				else:
					DDF = dd.concat([ DDF,  dd.from_pandas(df , chunksize = 200) ] , interleave_partitions=True)
			DFdict={}

	if verbose == True:
		print(df)
	return DDF

sspath = './SSdataset/'
fastas = glob.glob(sspath + '*.fasta')

print(fastas)
ddf = fastasToDF( fastas=fastas, verbose=True)

#turn fastas intoddf

#dump ddf to disk hdf5

#reload from disk

#learn keras model char to char
    #stateful?
    #embedding?
    
