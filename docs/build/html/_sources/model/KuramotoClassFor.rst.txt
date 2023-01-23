Kuramoto model
===================================

Implements the Kuramoto model in three different versions.
The instatiated model already includes the method for its integration called by the method **simulate()**.

-----------------
Clasical Kuramoto
-----------------
.. math::

   \dot{\theta}_{n}(t)=\omega_{n}+\sum_{n} c_{nm} \sin{(\theta_{m}(t)-\theta_{n}(t))}
   

-----------------
Delayed Kuramoto
-----------------
.. math::

   \dot{\theta}_{n}(t)=\omega_{n}+\sum_{n} c_{nm} \sin{(\theta_{m}(t-\tau_{mn})-\theta_{n}(t))}


----------------------------------
Partially Forced Delayed Kuramoto
----------------------------------
.. math::

   \dot{\theta}_{n}(t)=\omega_{n}-\delta_{n} F\sin{(\sigma t-\theta_{n})}+\sum_{n} c_{nm} \sin{(\theta_{m}(t-\tau_{mn})-\theta_{n}(t))}
   
An online tutorial could be found in `Tutorial Kuramoto model <https://colab.research.google.com/github/FelipeTorr/KuramotoNetworksPackage/blob/main/KuramotoNotebook.ipynb>`_.


.. autoclass:: model.KuramotoClassFor.Kuramoto
   :members:
   :show-inheritance:
