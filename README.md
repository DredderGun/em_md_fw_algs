## Problem statement

We aim in this project to compare the EM algorithm with classical convex optimization algorithms, such as:

* the Frank–Wolfe algorithm with a diminishing step size $\frac{1}{k + 1}$;
* the Frank–Wolfe algorithm with the shortest step size;
* the Frank–Wolfe algorithm with the shortest step size with Bragman divergence;
* the Mirror Descent algorithm.

We minimize a topic modeling problem with a decorrelation regularizer:  
$$  
\begin{align*}
& \underset{\Theta, \Phi}{\text{min}}
& & f(\Theta, \Phi) = -\sum_{d \in D} \sum_{w \in W} n_{dw} \ln{(\phi_{wt} \theta_{td})} + \frac{\tau}{2} \sum_{t \in T} \sum_{s \in T / t} \sum_{w \in W} \phi_{ws} \phi_{wt} \
& \text{s.t.}
& & \sum_{w \in W} \phi_{wt} = 1, ; \phi_{wt} \geq 0 \
& & & \sum_{w \in W} \theta_{wt} = 1, ; \theta_{wt} \geq 0
\end{align*} \quad \quad \quad (1)  
$$  

where $\Theta \in \mathbb{R}^{|T| \times |W|}$ and $\Phi \in \mathbb{R}^{|D| \times |T|}$. Here, $|T|$ is the number of topics (set manually by the user), $|W|$ is the number of words in the dictionary, and $|D|$ is the number of documents. Each document is represented as a bag-of-words object (e.g., a web page, a text document, or another text source). Note that in both matrices all rows sum to 1; that is, the feasible set is a productinc. product of simplices.

For more details, see *“Rethinking Probabilistic Topic Modeling from the Point of View of Classical Non-Bayesian Regularization”* by Konstantin Vorontsov, or I. A. Irkhin and K. V. Vorontsov, *“Convergence of the Algorithm of Additive Regularization of Topic Models”* (in Russian).

Experiments were conducted on the Koz dataset (taken from [https://docs.bigartm.org/en/stable/tutorials/datasets.html](https://docs.bigartm.org/en/stable/tutorials/datasets.html)).

## Structure

The results can be found in the Jupyter Notebook `topic_modeling_alg_testing/ipnyb/EM_FW.ipynb`.
