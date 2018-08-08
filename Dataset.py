#test functions
import functions
import config
import os
import glob
from dask.multiprocessing import get
import dask
import gc

from matplotlib import pyplot as plt

from distributed import Client, progress
from multiprocessing.pool import ThreadPool

if config.distributed == False:
	dask.set_options(pool=ThreadPool(functions.mp.cpu_count() *4 ))
	dask.set_options(get=dask.threaded.get)
else:
	if __name__ == '__main__':
	#create a scheduler and workers
		client = Client()

if __name__ == '__main__':
	#client = Client(config.clusterfile)
	if config.create_data == True and  __name__ == '__main__':
		chunks = functions.mp.cpu_count()
		window = functions.sig.gaussian(config.nGaussian, std=config.stdv)
		window /= functions.np.sum(window)

		propdict = functions.loadDict(config.proptable)
		hyperparams={'propdict': propdict  , 'Gaussian':window , 'clipfreq':400, 'verbose' : config.verbose , 'printResult' : False , 'onlycount' : False  }

		#####configure pipelines #######################
		physical_pipeline_functions = [ functions.seq2numpy,  functions.seq2vec , functions.gaussianSmooth, functions.fftall , functions.clipfft ]
		configured = []

		for func in physical_pipeline_functions:
			configured.append(functions.functools.partial( func , hyperparams=hyperparams ) )
		physicalProps_pipeline  = functions.compose(reversed(configured))

		phobius_pipeline_functions = [  functions.runphobius,  functions.parsephobius, functions.gaussianSmooth, functions.fftall , functions.clipfft ]
		configured = []
		for func in phobius_pipeline_functions:
			configured.append(functions.functools.partial( func , hyperparams=hyperparams ) )
		phobius_pipeline = functions.compose(reversed(configured))


		garnier_pipeline_functions = [  functions.runGarnier,  functions.GarnierParser, functions.gaussianSmooth, functions.fftall , functions.clipfft ]
		configured = []
		for func in garnier_pipeline_functions:
			configured.append(functions.functools.partial( func , hyperparams=hyperparams ) )
		garnier_pipeline = functions.compose(reversed(configured))

		##### final functions to be mapped #####
		applyGarniertoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=garnier_pipeline  , hyperparams=hyperparams )
		applyphobiustoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=phobius_pipeline  , hyperparams=hyperparams )
		applyphysicaltoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=physicalProps_pipeline  , hyperparams=hyperparams )

		#use pipelines to generate feature vecotrs for fastas
		#store matrices in hdf5 files

		pipelines={'physical':applyphysicaltoseries ,  'garnier': applyGarniertoseries , 'phobius': applyphobiustoseries }
		inputData = {'physical':'seq', 'phobius': 'fasta' , 'garnier' : 'fasta' }
		df = None

		print('done configuring pipelines')

		######## load fasta into dataframe ############################################
		if config.RunWithEcod==True:
			print('learning on protein structure clusters')
			fastas = [config.ecodFasta]
			df = functions.fastasToDF(fastas, config.verbose , config.RunWithEcod )
			if config.verbose == True:
				print(df)
			df['Y'] = df['ECOD domain'].map( lambda x : x.split('.')[0] )
			gc.collect()
		else:
			positives = [x[0] for x in os.walk(config.positive_datasets)]
			if config.generate_negative == True:
				print('generating negative sample fasta')
				fastaIter = functions.SeqIO.parse(config.uniclust, "fasta")
				sample = functions.iter_sample_fast(fastaIter, config.NegSamplesize)
				samplename = config.negative_dataset+str(config.NegSamplesize)+'rand.fasta'
				with open(samplename, "w") as output_handle:
					functions.SeqIO.write(sample , output_handle , format = 'fasta')
			dfs=[]
			for folder in positives + [config.negative_dataset]:
				fastas = glob.glob(folder+'/*fasta')
				print(folder)
				if 'ECOD'not in folder:
					print(fastas)
					if len(fastas)>0:
						df = functions.fastasToDF(fastas, config.verbose)
						df['Y'] = functions.np.string_(folder)
						dfs.append(df)
			df= functions.dd.concat( dfs , axis =0 , interleave_partitions=True )

		if config.RunWithEcod==True:
			df.set_index('ECOD uid')
		else:
			df.set_index('desc')

		#print('loaded fastas with categories:')
		#print( len(df['Y'].unique().compute(get=get ) ) )
		#print(len(df))

		if config.visualizeOne ==True:

			dfs = df.to_delayed()
			data = dfs[0].iloc[0]['fasta'].compute()
			print(data)
			outputs = []
			out = data
			for i,pipeline in enumerate(configured):

				out = pipeline(out)
				outputs.append(out)
				print(i)
				print(out)
				if i > 0 :
					for j in range(out[0].shape[0]):
						plt.plot(out[0][j,:].ravel())
						plt.show()
						plt.savefig('./fig'+str(i)+'.png')


	################################# run pipelines on sequences ###########################
		for name in config.pipelines:
			print('calculating ' + name)
			meta = functions.dd.utils.make_meta( {name: object }, index=None)
			df[name] = df[inputData[name]].map_partitions( pipelines[name] ).compute(get=get)

		if config.verbose == True:
			print(df)



	##############################save###########################################################3
		if config.save_data_tohdf5 == True:
			if config.shards == True:
				dfs = df.to_delayed()
				#save datsets to hdf5 format in chunks
				filenames=[]
				if config.overwrite == True:
					print('overwriting hdf5 datasets')
					for i in range(len(dfs)):
						name = config.savedir + config.dataname+str(i)+'.hd5'
						filenames.append(name)
				else:
					print('appending hdf5 datasets')
					filenames = glob.glob(config.savedir + '*.hd5')
				inputlist = list(zip(dfs, filenames))
				writes = [functions.delayed(functions.hd5save)(df , fn , config.overwrite , config.verbose, config.pipelines ) for df, fn in inputlist]
				functions.dd.compute(*writes , get=get)
			else:
				#not tested...
				df.to_hdf( config.savedir+ config.dataname + '.hd5' , key = '/'  , format ='table' )
			##################################load ###############################################################
		print('DONE')
