__version__ = "0.0.1"

try:
	#Model
	import model.KuramotoClassFor
	import model.parserConfig
	#analysis
	import analysis.frequency
	import analysis.synchronization
	import analysis.cluster
	import analysis.connectivityMatrices
	import analysis.Wavelets
	#simulations
	from simulations.testModel import RunKuramotoFor
	from simulations.testModelConfig import RunKuramotoFor
	 
	#plot
	import plot.circular
	import plot.animation
	import plot.networks
	import plot.scatter
	import plot.video_twonodes 
except ModuleNotFoundError:
	#To avoid error when using in Google Colab
	#Model
	import KuramotoNetworksPackage.model.KuramotoClassFor
	import KuramotoNetworksPackage.model.parserConfig
	#analysis
	import KuramotoNetworksPackage.analysis.frequency
	import KuramotoNetworksPackage.analysis.synchronization
	import KuramotoNetworksPackage.analysis.cluster
	import KuramotoNetworksPackage.analysis.connectivityMatrices
	import KuramotoNetworksPackage.analysis.Wavelets
	#simulations
	from KuramotoNetworksPackage.simulations.testModel import RunKuramotoFor
	from KuramotoNetworksPackage.simulations.testModelConfig import RunKuramotoFor
	 
	#plot
	import KuramotoNetworksPackage.plot.circular
	import KuramotoNetworksPackage.plot.animation
	import KuramotoNetworksPackage.plot.networks
	import KuramotoNetworksPackage.plot.scatter
	import KuramotoNetworksPackage.plot.video_twonodes 
