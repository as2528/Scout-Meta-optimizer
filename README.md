## Scout Meta-optimizer

### Quick intuition
Most first-order optimisers are greedy: they step exactly in the current gradient direction.  
`Scout` interrupts that every **N** iterations, peeks a few steps into two candidate directions, then hands the base optimiser a *replacement* gradient pointing along the more promising path.

### Notation
* $\theta \in \mathbb{R}^{d}$ – parameters  
* $\mathcal{L}(\theta)$ – minibatch loss  
* $\mathbf{g} = \nabla_{\theta}\mathcal{L}(\theta)$ – current gradient  
* $\hat{\mathbf{g}} = \mathbf{g}/\|\mathbf{g}\|$ – unit gradient  
* $\hat{\mathbf{r}}$ – random unit vector orthogonal to $\hat{\mathbf{g}}$

### Algorithm (one scout cycle)
1. **Directions to test**  
   $\mathcal{D} = \{\hat{\mathbf{g}},\,\hat{\mathbf{r}}\}$  
   Draw $\hat{\mathbf{r}}$ by sampling a Gaussian vector and orthogonally projecting.
2. **Probe each direction**  
   * Start from $\theta^{(0)} = \theta - \rho\,\hat{\mathbf{d}}$ (small radius $\rho$).  
   * Roll out $k$ vanilla GD steps with learning-rate $\eta$:  
     $$\theta^{(t+1)} = \theta^{(t)} - \eta\,\nabla\mathcal{L}(\theta^{(t)})$$
   * Record the best loss observed,  
     $$F(\hat{\mathbf{d}}) = \min_{0 \le t \le k}\mathcal{L}(\theta^{(t)})$$
3. **Pick winner**  
   $$\hat{\mathbf{d}}^{\star} = \arg\min_{\hat{\mathbf{d}} \in \mathcal{D}} F(\hat{\mathbf{d}})$$
4. **Replace gradient**  
   $$\mathbf{g}_{\text{scout}} = \|\mathbf{g}\| \; \hat{\mathbf{d}}^{\star}$$
   Pass $\mathbf{g}_{\text{scout}}$ to Adam/SGD/whatever instead of $\mathbf{g}$.

### Hyper-parameters (defaults that worked on CIFAR-10)
| name            | meaning                    | typical |
|-----------------|----------------------------|---------|
| `scout_every` N | probe frequency            | 100     |
| `scout_radius` $\rho$ | initial offset           | $5\times10^{-4}$ |
| `lookahead_steps` k | rollout length            | 1       |

### Overhead
Two probe directions every `scout_every=N` steps cost  
$$\text{extra\_F/B} \;=\; \frac{2\,(k+1)}{N}$$  
forward/backward passes.  
With the defaults above that’s about **2 %** extra compute.

### Why it sometimes helps (no theorems yet)
* **Saddle evasion** – orthogonal poke finds downhill direction when $\|\mathbf{g}\|$ is near 0.  
* **Shallow-pit escape** – roll-out filters gradients that immediately raise loss.  
* **Cheap diversity** – two directions give most of the benefit; more would snowball cost.

### Limitations
* Adds compute and memory traffic.  
* Only one random orthogonal direction; broader sampling might be better but isn’t free.  
* Zero convergence theory so far—purely empirical.
