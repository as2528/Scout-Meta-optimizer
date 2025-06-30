# Scout-Meta-optimizer
\section{Scout: Gradient Peeking in Rugged Landscapes}

Let $\theta\in\mathbb{R}^{d}$ be the parameter vector,  
$\mathcal{L}(\theta)$ the minibatch loss, and
$\mathbf{g}=\nabla_{\theta}\mathcal{L}(\theta)$ its gradient.

\subsection{High-level idea}
Most first-order methods chase $\mathbf{g}$ greedily; they can buckle at
saddles or get hypnotised by shallow minima.  
\textbf{Scout} pauses every $N$ steps to \emph{peek} down two directions:

1.  the normalised gradient $\hat{\mathbf{g}}=\mathbf{g}/\lVert\mathbf{g}\rVert$  
2.  a random unit vector $\hat{\mathbf{r}}$ drawn uniformly from the
    subspace orthogonal to $\hat{\mathbf{g}}$.

A short look-ahead rollout (length $k$) is executed along each direction.
Whichever rollout attains the lower loss becomes the steering direction for
this iteration; its magnitude is rescaled back to $\lVert\mathbf{g}\rVert$ so
the outer optimiser (SGD, Adam, RMSProp, …) keeps control of step size.

\subsection{Mathematical sketch}

\begin{enumerate}
\item \textbf{Gradient.}\;
      $\mathbf{g}=\nabla\mathcal{L}(\theta)$,
      $\hat{\mathbf{g}}=\dfrac{\mathbf{g}}{\lVert\mathbf{g}\rVert+\varepsilon}$.
\item \textbf{Orthogonal probe.}\;
      Draw $\mathbf{u}\sim\mathcal{N}(0,I_{d})$ and project:
      \[
        \hat{\mathbf{r}}
        =\frac{\mathbf{u}-\langle\mathbf{u},\hat{\mathbf{g}}\rangle\hat{\mathbf{g}}}
               {\lVert\mathbf{u}-\langle\mathbf{u},\hat{\mathbf{g}}\rangle\hat{\mathbf{g}}\rVert}.
      \]
\item \textbf{Roll-outs.}\;
      For each $\hat{\mathbf{d}}\in\{\hat{\mathbf{g}},\hat{\mathbf{r}}\}$ set
      $\theta^{(0)}=\theta-\rho\,\hat{\mathbf{d}}$ and iterate
      \[
        \theta^{(t+1)}=\theta^{(t)}-\eta\,\nabla\mathcal{L}(\theta^{(t)}),
        \quad t=0,\dots,k-1.
      \]
      Record $F(\hat{\mathbf{d}})=\min_{0\le t\le k}\mathcal{L}(\theta^{(t)})$.
\item \textbf{Direction choice.}\;
      $\hat{\mathbf{d}}^{\star}=\arg\min F(\hat{\mathbf{d}})$.
\item \textbf{Gradient replacement.}\;
      Use
      $\displaystyle
        \mathbf{g}_{\text{scout}}
        =\lVert\mathbf{g}\rVert\,\hat{\mathbf{d}}^{\star}
      $
      in place of $\mathbf{g}$ for the upcoming optimiser step.
\end{enumerate}

\subsection{Computational cost}
One scout cycle costs $(k+1)$ extra forward/backs per probe direction; with two
directions every $N$ steps the amortised overhead is
$\;\frac{2(k+1)}{N}\;$ relative to vanilla training.  In CIFAR-10 experiments
we used $N{=}100$, $\rho\!=\!5\!\times\!10^{-4}$, $k{=}1$
($\sim$2 % extra compute) and saw repeatable $\!+\!(0.5$–$1)$-point top-1
accuracy bumps in early epochs.  Gains flatten later—as expected once both
optimisers reach the same broad basin.

\subsection{Why it can help (intuition, not proof)}
\begin{itemize}\setlength\itemsep{2pt}
  \item \textbf{Saddle evasion.}\; At strict saddles $\mathbf{g}=0$ but the
        Hessian has mixed signs; a small orthogonal poke often finds a downhill
        slope that plain GD would miss until numerical noise rescues it.
  \item \textbf{Shallow-pit escape.}\; Short roll-outs filter out directions
        that immediately worsen loss even if the local gradient points that
        way, reducing the chance of drifting into flat, unproductive pockets.
  \item \textbf{Cheap ensemble search.}\; Two directions strike a pragmatic
        balance—enough diversity to dodge obvious traps without exploding the
        batch-time budget.
\end{itemize}

\subsection{Hyper-parameters (defaults that worked for us)}
\[
N=100,\quad \rho=5\times10^{-4},\quad k=1.
\]

\vspace{1ex}
That’s it: Scout is nothing more exotic than “look before you leap, once in a
while.”  If you need formal guarantees, go prove them—right now all we have is
empirical evidence and a healthy dose of skepticism about easy wins in
optimisation.

% --- End of math-only section -----------------------------------

