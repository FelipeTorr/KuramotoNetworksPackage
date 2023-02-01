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
	import KuramotosNetworksPackage.model.KuramotoClassFor
	import KuramotosNetworksPackage.model.parserConfig
	#analysis
	import KuramotosNetworksPackage.analysis.frequency
	import KuramotosNetworksPackage.analysis.synchronization
	import KuramotosNetworksPackage.analysis.cluster
	import KuramotosNetworksPackage.analysis.connectivityMatrices
	import KuramotosNetworksPackage.analysis.Wavelets
	#simulations
	from KuramotosNetworksPackage.simulations.testModel import RunKuramotoFor
	from KuramotosNetworksPackage.simulations.testModelConfig import RunKuramotoFor
	 
	#plot
	import KuramotosNetworksPackage.plot.circular
	import KuramotosNetworksPackage.plot.animation
	import KuramotosNetworksPackage.plot.networks
	import KuramotosNetworksPackage.plot.scatter
	import KuramotosNetworksPackage.plot.video_twonodes 
