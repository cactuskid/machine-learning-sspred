import config
import functions
import keras_model
from matplotlib import pyplot as plt
import random
from dask_ml.preprocessing import StandardScaler,DummyEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import glob
import numpy as np
import tensorflow as tf
import dask
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
from sklearn.model_selection import train_test_split
#create dask arrays for X and Y to feed to keras

from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from keras import backend as K

from distributed import Client, progress
import distributedTensorFlow as dtf 

from sklearn.utils import shuffle
print('loading dataset')
filenames = glob.glob(config.savedir + config.dataname +'*.hd5')
print(filenames)
dataset = functions.DaskArray_hd5loadDataset(filenames , verbose=config.verbose)
print('dataset loaded with keys:')
print(dataset.keys())
Y= dataset['Y']
data = []
for pipeline in dataset.keys():
	if pipeline in config.pipelines:
		data.append( dataset[pipeline] )
		if config.visualize == True:
			for j in range(5):
				i = random.randint( 0, dataset[pipeline].shape[0] )
				plt.title(pipeline + str(Y[i].compute()))
				plt.plot(dataset[pipeline][i,:].compute())
				plt.save_fig( './figures/'+pipeline + str(Y[i].compute()+'.png'))
X = functions.da.concatenate(data, axis = 1)

######################################make NN ################################################################
scaler = StandardScaler( copy = True )
le = LabelEncoder()
enc = OneHotEncoder(sparse = False)
epochs = 20
functions.np.random.seed(0)
print(Y[0:10].compute() )
encY = le.fit_transform(Y.ravel())
dummy_y = enc.fit_transform(encY.reshape(-1, 1) )

print(dummy_y)
print('scaling X')
if config.scaleX == True:
	X = scaler.fit_transform( X )
print('Done')
inputdim=X.shape[1]
outputdim = dummy_y.shape[1]
print(inputdim)
print(outputdim)

retmodel = keras_model.configure_model(keras_model.selu_network, inputdim , outputdim  )


if config.distributed == True:
	client, dask_cluster,  tf_spec, dask_spec = dtf.setup_tensorflow_cluster(clustertype='local' , TFservers= None, scale_up = False)	
	
	dtf.configure_cluster( client, dask_cluster , tf, ds  , retmodel , verbose)
	#start up worker processes
	#train model in distributed mode
	for i in range(epochs):
		distibutedTensorFlow.trainbatches(X,Y,dask_spec)
		

else:
	results = []

	device_name = "/cpu:0"
	with tf.device(device_name):
		x, y  = retmodel()
		opt = tf.train.AdamOptimizer
		labels = tf.placeholder(tf.float32, shape=(None, outputdim))

		encode_y = tf.argmax(y, axis = 1)
		encode_labels = tf.argmax(labels, axis=1)
		confusion = tf.confusion_matrix(encode_labels,encode_y)
		acc_value = accuracy(labels, y)

		loss = tf.reduce_mean(categorical_crossentropy(labels, y))
		#train_step = opt.minimize(loss)
		train_step = tf.train.AdamOptimizer().minimize(loss)
		sess = tf.Session()
		
		saver = tf.train.Saver()
		init_op = tf.global_variables_initializer()
		
		sess.run(init_op)

		X_train, X_test, y_train, y_test = train_test_split( X, dummy_y , test_size=0.15, random_state=42)
		with sess:
			try:
				pass
				#saver.restore(sess, "./model.ckpt")
			except:
				pass
			for k in range(epochs):
				save_path = saver.save(sess, "./model.ckpt")
				K.set_session(sess)
				train_step.run(feed_dict={x:X_train,
										  labels: y_train,
										  K.learning_phase(): 1})
				
				print('eval accuracy')
				acc = acc_value.eval(feed_dict={x: X_test,
											labels: y_test ,
											K.learning_phase(): 0})
				
				print(np.mean(acc))
				cm = confusion.eval( feed_dict={x: X_test,
											labels: y_test ,
											K.learning_phase(): 0} )

				print(cm)
				functions.plot_confusion_matrix(cm, le.classes_, normalize=False, title= str(k)+'Confusion matrix', cmap=plt.cm.Blues , outdir = './')
				functions.plot_confusion_matrix(cm, le.classes_, normalize=True, title= str(k)+'Confusion matrix Normalized', cmap=plt.cm.Blues , outdir = './')

###############################################wisdom ###########################################################
if config.visualize_model == True:
	pass 
	#show model output at different layes to learn something from the features
	