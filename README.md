# KuramotoNetworksPackage
Kuramoto model of coupled oscillators for brain networks 

The partially forced Kuramoto model of coupled oscillators, in addition to heterogenous connection strengths can include heterogeneous connection delays. 

$$
\dot{\theta}_n(t)=\omega_{n}+ F\delta_{n,C}sin(\theta_n+\sigma t) +\epsilon_{i}(t)+\frac{K}{N} \sum_{m=1}^{N} W \sin \left(\theta_{m}(t-\tau_{mn})-\theta_{n}(t-\tau_{nn})\right),
\label{eq:ForcedKuramotoNoise}
$$

## Project Structure

- The folder **model** contains the source code of the model(s) employed for all simulations.
- The folder **simulations** contains the configuration files to run simulations.
- The folder **test** contains test code.
- The folder **analysis** contains digital signal processing, synchronization, and network analysis tools.
- The folder **input_data** contains the connection matrices and other input_data
- The simulation result goes to the folder **timeseries_output**. Not here because their large size.
- The figures of the analysis and signal processing are inside the folder **figures**.
 

## TODO list
- Setters of model parameters
- Use a configuration .txt file to run simulations
- Separate utility and plot functions
- Document and comment
