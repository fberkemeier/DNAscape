Unified mapping framework
=========================

Mapping initiation rate, fork speed, replicated fraction and mean replication
timing uses one forward model. Inverse maps change the unknown kinetic field;
the equations predicting the observations remain the same. See
:doc:`mappingfunctions` for the app controls and :doc:`foundations` for units
and experimental interpretation.

One forward problem
-------------------

Let :math:`I(x,t)\geq0` denote potential origin firing per unit unreplicated
length and time, and let :math:`v(x,t)>0` denote a prescribed fork-speed field.
Forks travel in opposite directions according to
:math:`d\xi/dt=\pm v(\xi,t)`. For an observation at :math:`(x,t)`, the past
replication cone :math:`\Lambda_{x,t}[v]` contains precisely the earlier
potential origins whose forks could reach that observation.

For independent Poisson initiation attempts, the probability that this cone
contains no event is

.. math::

   H(x,t)=\iint_{\Lambda_{x,t}[v]} I(\xi,\tau)\,d\xi\,d\tau,
   \qquad S(x,t)=e^{-H(x,t)},
   \qquad f(x,t)=1-e^{-H(x,t)}.

:math:`H` is the cumulative replication hazard, :math:`S` is survival of
unreplicated DNA, and :math:`f` is the replicated fraction across cells.
Initiation attempts in already replicated DNA are suppressed; they do not
need to be simulated explicitly to calculate these ensemble observables.
This cone construction underlies the inhomogeneous KJMA description
[KineticBaker2014]_.

Mean timing is the survival integral

.. math::

   \overline T(x)=\int_0^\infty e^{-H(x,t)}\,dt.

Thus predicting timing and fraction is one calculation with two observation
operators. A finite time grid must capture the survival tail. If a model
retains a nonzero probability of never replicating, its unconditional mean
timing is infinite; truncating the integral cannot turn it into a finite
mean. Fraction observations may legitimately stop before completion; an
auxiliary survival integral over that observed window is a restricted mean,
rather than the unconditional mean timing.

The continuous equations use initiation density in
:math:`\mathrm{kb}^{-1}\mathrm{min}^{-1}`. The app's per-bin initiation rate
is divided by the spatial bin width at the Python mapping boundary. Fork
speed remains in kb/min. Internal numerical fields are time by space; public
space-time profiles retain the position-by-time convention described in
:ref:`concept-grids`.

Equivalent differential form
----------------------------

On a line, for smooth positive :math:`v`, the same cone integral satisfies

.. math::

   \partial_t\!\left(\frac{\partial_t H}{v}\right)
   -\partial_x\!\left(v\,\partial_xH\right)=2I,
   \qquad H(x,0)=\partial_tH(x,0)=0.

This is valid for a fully space-time-dependent deterministic speed. One way
to see the equivalence is to introduce the directional cone-edge fields

.. math::

   a=\frac{H_t}{v}+H_x,\qquad b=\frac{H_t}{v}-H_x.

The differential equation and compatibility of mixed derivatives give

.. math::

   a_t-\partial_x(va)=2I,\qquad
   b_t+\partial_x(vb)=2I,\qquad
   H_t=\frac{v}{2}(a+b).

These are two conservative transport equations with a common source. Their
characteristics are the edges of the replication cone. For constant speed,
the hazard equation reduces to :math:`H_{tt}-v^2H_{xx}=2vI`.
For a finite linear molecule without incoming forks, the incoming
characteristic conditions are :math:`b=0` at the left boundary and
:math:`a=0` at the right boundary. A cropped experimental region can instead
receive forks from outside; retain surrounding input data when those
contributions matter [KineticGauthier2012]_.

A particularly useful numerical shortcut takes a spatial primitive of the
source, :math:`F(x,t)=\int_{x_0}^x I(y,t)\,dy`, and transports two potentials:

.. math::

   (P_+)_t+v(P_+)_x=F,\qquad
   (P_-)_t-v(P_-)_x=F,\qquad
   H=P_--P_+.

Each potential accumulates :math:`F` along one backward characteristic; their
difference is the source integral between the two cone edges. A time step
updates two field arrays instead of retracing every complete past cone.
With the primitive anchored at the left boundary and no exterior origins,
the incoming potential values are
:math:`P_+(x_{\mathrm L},t)=0` and
:math:`P_-(x_{\mathrm R},t)=\int_0^t F(x_{\mathrm R},\tau)\,d\tau`;
both potentials start at zero. These values retain the primitive's offset
without introducing incoming replication forks.
Stationary initiation and constant speed also admit direct integration of
the triangular travel-time kernel within each spatial cell.

The integral and differential descriptions are mathematical alternatives,
not different replication models. Efficient numerical transport or potential
calculations are used where their domain and boundary assumptions apply;
retarded characteristic boundary values supply the finite-influence and
periodic calculation without retracing every past cone. Deterministic kinetic calculations avoid repeatedly
sampling individual S phases during optimisation [KineticGauthier2014]_.
The actual discretisation is differentiated during
fitting, so its prediction and gradient refer to the same numerical model.

Influence radius and periodic regions
-------------------------------------

The default influence radius is **2000 kb**, adjustable in Mapping. At a
linear position, a finite radius restricts contributing origins to
:math:`[x-R,x+R]`, intersected with the available region. It limits the
spatial reach of the calculation, not the resolution or output interval.

This is an explicit approximation to unrestricted KJMA when a physical cone
extends beyond the radius. It is exact over a finite time horizon whenever
the fastest possible fork cannot travel farther than :math:`R`. Its effect
should be checked by increasing the radius for late-replicating or
origin-poor regions. For homogeneous constant inputs, away from the ends,

.. math::

   H_R(t)=
   \begin{cases}
      Iv t^2, & t\leq R/v,\\
      2IRt-IR^2/v, & t>R/v.
   \end{cases}

On a periodic region of circumference :math:`L`, sources are counted
**once**, even when backward paths have wrapped around the region. A radius
of at least :math:`L/2` includes the whole circular genomic neighbourhood.
For constant inputs and that whole neighbourhood,

.. math::

   H(t)=
   \begin{cases}
      Iv t^2, & t\leq L/(2v),\\
      ILt-IL^2/(4v), & t>L/(2v).
   \end{cases}

A wave equation solved with ordinary periodic boundary conditions would
continue counting periodic copies after the cone covers the circle. The
unique-source cone construction avoids that error and introduces no special
fit preference at the periodic join. Smaller circular masks wrap across the
join using the same genomic neighbourhood rule.

Fast forward calculations
~~~~~~~~~~~~~~~~~~~~~~~~~

The default backend selects a shortcut from the actual supplied dependences:

* Stationary initiation and stationary speed use exact centred-cell integration
  in the travel coordinate :math:`Y(x)=\int dx/v(x)`. Within the permitted
  neighbourhood, the kernel is
  :math:`(t-|Y(x)-Y(\xi)|)_+`. Primitives of :math:`I` and :math:`IY` evaluate
  its integral without stepping characteristics. The reverse pass differentiates
  those same cell integrals. This shortcut also fits spatial speed directly.
* Stationary initiation with spatially uniform, time-dependent speed uses one
  spatial convolution per observation time. With :math:`D(t)=\int_0^t v(u)du`,
  a source at distance :math:`r` can fire until
  :math:`\sigma=D^{-1}((D(t)-r)_+)`. Integrating :math:`\sigma` over a source
  cell uses :math:`J(s)=\int_0^s u v(u)du`. This integrates the prescribed
  piecewise-linear speed exactly and supports a discrete speed adjoint, so
  spatial-initiation and temporal-speed fits share the same faster calculation.
* Constant speed and a temporal initiation field use a causal space-time
  convolution. Its kernel integrates both the source cells and linear temporal
  interpolation exactly, including the partial temporal hat at zero. FFTs
  reuse that kernel across initiation-fit proposals; the radius limits its
  spatial support.
* Spatially uniform kinetics use three temporal primitives. Write
  :math:`D=\int v\,dt`, :math:`A=\int I\,dt` and :math:`B=\int ID\,dt`.
  A one-sided cone of permitted length :math:`d` contributes

  .. math::

     G(d,t)=dA(\sigma)+D(t)[A(t)-A(\sigma)]-B(t)+B(\sigma),

  where :math:`D(\sigma)=\max(0,D(t)-d)`. The left and right contributions
  add to the hazard. Within-interval polynomials are integrated analytically.
  The derivative through the crossing time cancels, making this also a cheap
  speed-fitting operator.
  Uniform-input fits evaluate each distinct boundary history once, combining
  the weights of equivalent observed positions in the objective and adjoint.
  This preserves the full data MSE, including for mean-fraction observations.
* General fields use two transported potentials. Positive travel coordinates
  :math:`Y=\int dx/\bar v` invert each characteristic step without a spatial
  CFL restriction. The numerical speed representation interpolates slowness
  between samples; time-averaged velocities and two-point Gauss source
  quadrature provide the temporal approximation.

For the general radius correction, let
:math:`C(x,t)=\int_0^tF(x,\tau)d\tau` and :math:`Q_\pm=P_\pm-C`.
Advected monotone characteristic labels locate the earlier times
:math:`\sigma_\pm` when the paths crossed :math:`x\mp R`. Then

.. math::

   H_R(x,t)=H(x,t)+Q_+(x-R,\sigma_+)-Q_-(x+R,\sigma_-).

A correction is zero until its boundary is reached. On a circle use
:math:`R\leq L/2`; the two contributions then cover every source at most once.
This avoids the previous quadratic-in-time cone replay. Integer-bin radius
shifts use array rolls; other shifts interpolate. Uniform speed has a shared
crossing clock and avoids storing spatial characteristic labels.

The forward approximation retains non-negative, non-decreasing hazards using
an explicit monotone projection when needed. Its active branches are included
in the discrete adjoint. Analytical cases, independent cell quadrature and
refinement checks distinguish exact shortcuts from the general numerical
approximation. The older cone operator remains a reference backend.
Initiation fits with prescribed speed cache departure positions separately
from source interpolation geometry. Speed fits retain reverse derivatives
directly to avoid decoding them during each reverse pass. Both fork directions
share the travel coordinate for a time slice, and stationary source primitives
are shared across time. The caches retain their fixed memory budgets.

Fitting kinetic fields
----------------------

To infer initiation rate, supply speed and either timing or replicated
fraction. To infer speed, supply initiation rate and either observation.
All constant, spatial, temporal and space-time input combinations use the
same observation operators.

Constant targets have one fitted amplitude. Spatial targets have an amplitude
at each genomic sample. Temporal and space-time targets use a shifted
Weibull-hazard-shaped basis,

.. math::

   h_k(t;d)=k\left(
      \frac{\max(t-d,\tau_0)}{t_{\mathrm{ref}}}
      \right)^{k-1}
   \quad (t\geq d),

with zero initiation contribution before the delay. The positive reference
time is fixed from the observation, so numerical time resolution does not
define the model's time scale. Fitted shapes range from 0.25 to 8, allowing
decreasing or increasing temporal trends. The positive physical onset scale
:math:`\tau_0` defaults to one minute and keeps the first part of the basis
finite, rather than introducing a time-step-dependent shift. The fitted
fields are

.. math::

   I(x,t)=\alpha(x)h_k(t;d(x)),\qquad
   v(x,t)=\beta(x)\,[c+h_k(t;d(x))],\quad c>0.

Here the dimensionless basis leaves initiation amplitude :math:`\alpha` in
the units of initiation density, and speed amplitude :math:`\beta` in kb/min.
The same log-amplitude fitting procedure applies to either unknown field.
For time-only targets, amplitude and :math:`k` are shared across the
region. For space-time targets, both can vary with position. Local shapes
:math:`k(x)` allow different temporal trends in different regions, rather
than forcing the whole field to be a product of one spatial and one temporal
profile. A temporal step changes the global amplitude and, when enabled,
the Weibull shape or local shapes; speed also fits its positive basal
component :math:`c`. Spatial steps change relative local amplitudes.
Supplied input fields may have arbitrary compatible dependence.
The positive amplitudes are parameterised logarithmically, with target bounds
applied to the resulting field.

These are deterministic kinetic parameterisations inspired by a Weibull
hazard. They do not assign independently random Weibull transit times to
forks. They also restrict the available space-time shapes: an arbitrary
matrix cannot generally be reconstructed within this family.

Both inverse problems minimise the discrepancy in the observed quantity:

.. math::

   \mathcal L(\theta)=
   \operatorname{mean}_{\mathcal V}
   \left[\frac{y_{\mathrm{model}}(\theta)-y_{\mathrm{observed}}}
   {q}\right]^2
   +\frac{\lambda_s}{2}
      \left[\operatorname{mean}\!\left((\Delta_x\log A)^2\right)
      +\operatorname{mean}\!\left((\Delta_x\log k)^2\right)\right]
   +\frac{\lambda_p}{2}\operatorname{mean}\!\left((\theta-\theta_0)^2\right).

Here :math:`\mathcal V` contains valid observed samples. The scale :math:`q`
is fixed from timing observations and is one for fraction observations;
:math:`A` is the initiation or speed amplitude. The remaining terms penalise
spatial roughness and displacement from the starting parameters.
Shape roughness applies when local shapes are
fitted. Periodic spatial roughness includes the join.
The displayed data MSE remains separate from these penalties.

The chain rule through :math:`f=1-e^{-H}` and the survival integral supplies
a hazard sensitivity. If :math:`g` is the derivative with respect to the
predicted observation, the two cases give

.. math::

   \frac{\partial\mathcal L}{\partial H_{j,i}}
   =g_{j,i}\,S_{j,i}
   \quad\hbox{(fraction)},\qquad
   \frac{\partial\mathcal L}{\partial H_{j,i}}
   =-g_i\,w_j\,S_{j,i}
   \quad\hbox{(mean timing)},

where :math:`w_j` are the temporal quadrature weights. A discrete adjoint
then propagates this sensitivity to initiation
samples and speed samples, including motion of the cone edges. This gives
all parameter derivatives in a reverse pass instead of one additional
forward map per fitted genomic value. It also avoids estimating noisy
second derivatives of measured :math:`-\log(1-f)` directly.

Spatial and temporal phases each default to **10 iterations** when
applicable. A space-time fit can alternate the two phases (**Cross fitting**)
or finish one before the other (**Space then time** or **Time then space**).
Large unmasked fits use a hierarchy of coarse grids, then refine and verify
on the selected spatial grid. This changes the optimisation path; output
resolution and the final reported residual use the native grid. Phase budgets
are shared across levels, and the fitting order is retained. Coarse-stage
errors are recorded separately from native-grid fit history.

Chromosome-sized fits retain the selected spatial and temporal resolution.
The native calculation is split into overlapping receiving regions, with
halos covering the influence radius and the transport interpolation history.
Only each region's central receivers contribute to the shared objective;
adjoints from overlapping source regions accumulate into one parameter vector.
Periodic halos wrap across the genomic join, and regional transforms keep
their global selection, interpolation anchors and frozen values. Large
space--time targets use temporary disk-backed arrays, released with the result,
rather than requiring every intermediate field to fit in RAM at once.
Late proposals can therefore cost more than early ones: they act on a finer
grid. Dependences can also require different completion horizons for the
survival integral; fewer fitted parameters do not guarantee a shorter fit.

**Adaptive fitting** can be disabled. **Fit tolerance** defaults to 0.001
(0.1% of timing, or 0.001 fraction); zero disables it. Local updates stop only
when the prediction is sufficiently close throughout their influence
neighbourhood. Global and local temporal parameters are scaled separately,
and line-search steps shrink as the residual decreases.
The same tolerance applies on coarse grids; converged regions are not updated
merely because another refinement level remains. Automatic initiation guesses
use local supplied speeds over the relevant replication-time window, rather
than later samples from a long upload. Spatial initiation updates also use the
exact common-amplitude sensitivity :math:`-\int H e^{-H}dt` to scale their
proposals. The full objective and its adjoint still determine acceptance.
Spatial amplitude updates for space-time speed fitted to timing use the
homogeneous unrestricted-cone scaling :math:`\overline T\propto v^{-1/2}` as a local
preconditioner. The coupled adjoint must agree with the proposed descent
direction, and the full line search governs acceptance. This can reach the
configured pointwise tolerance sooner, changing the optimisation path and
the distribution of residuals without relaxing that tolerance.
With a positive tolerance, a phase also stops after two negligible or rejected
updates; a useful update in the other phase can reactivate it. This avoids
repeating line searches on a plateau. Zero disables these convergence checks.

An iteration is a block proposal; line-search trial predictions are not
additional user-requested iterations. A proposed update is accepted only
when both the penalised objective and data MSE do not increase. Masked
initial target values are projected back at every evaluation and remain
fixed; supplied inputs are never fitted.

Progress has one first prediction and one iteration budget across all grids.
After that prediction, the counter includes completed spatial and temporal
updates; neither line-search trials nor refinement initialization count as
iterations. ETA estimates use measured proposal costs and include the pending
finer grids and their initialization. They are updated as those grids are
measured, and remain visible through final preview preparation.

What a fitted programme means
------------------------------

Timing alone removes much of the temporal information in :math:`f(x,t)`.
On an unbounded line with unrestricted influence, even homogeneous constant
kinetics illustrate the ambiguity:
:math:`\overline T=\sqrt{\pi}/(2\sqrt{Iv})` constrains a product, rather than
separately identifying initiation and speed. Supplying one kinetic input,
choosing a temporal family, setting bounds and using regularisation make a
fit tractable; they do not guarantee a unique biological reconstruction.
Inspect the predicted observation and residuals, and compare independent
measurements wherever possible. More iterations improve optimisation of the
chosen model, not the amount of experimental information
[KineticBaker2014]_.

Verification and benchmarking
------------------------------

Benchmarks cover 10,000 spatial samples at 1 kb across all sixteen kinetic
dependence pairs, fitting each kinetic target from timing and from full
space-time fractions. With the default ten updates per applicable phase,
the 32 timing fits took approximately **499 seconds before these optimisations
and 134 seconds afterwards**, a 3.7-fold reduction in aggregate cost.
For example, spatial initiation with temporal speed fell from 22.4 to
0.55 seconds, and temporal speed with spatial initiation from 9.8 to
0.60 seconds. These are local measurements; costs vary with the signals,
clocks, boundary conditions and line searches.

The complete genomic interval is retained. Automatic timing horizons expand
when the survival tail requires it; later samples in a supplied profile do
not force unnecessary late-time predictions. Coarse-grid fraction averages
are clipped to their valid probability range after validating the original
observations, avoiding rejection of saturated values rounded to
:math:`1+\epsilon`.

The fitted temporal family need not contain a supplied synthetic programme.
Residuals therefore measure observation fits, rather than unique recovery of
the input parameters. Individual residuals can change with the improved
initialisation and adaptive stopping. Final errors are checked on the native
observation grid.

Forward checks cover all sixteen dependence pairs and both boundary types.
Across those 32 examples, the largest change from the previous calculation
was 0.0000462 in fraction, or 0.00550 minutes in the survival integral.
Independent analytical hazards, source-cell/time quadrature, clipped radii,
probability monotonicity, finite-difference adjoints and native-grid replay
provide separate correctness checks. Kinematic-cache reuse preserves the
general forward calculation and both of its adjoints exactly.

References
-----------

.. [KineticBaker2014] A. Baker and J. Bechhoefer (2014).
   *Inferring the spatiotemporal DNA replication program from noisy data.*
   Physical Review E 89, 032703.
   `Primary paper <https://www.sfu.ca/chaos/assets/papers/2014/baker-pre14.pdf>`_.

.. [KineticGauthier2012] M. G. Gauthier, P. Norio and J. Bechhoefer (2012).
   *Modeling Inhomogeneous DNA Replication Kinetics.* PLOS ONE 7, e32053.
   `Primary paper <https://pmc.ncbi.nlm.nih.gov/articles/PMC3296702/>`_.

.. [KineticGauthier2014] M. G. Gauthier, A. Dube and J. Bechhoefer (2014).
   *Numerical modeling of inhomogeneous DNA replication kinetics.*
   `Preprint <https://arxiv.org/abs/1408.5216>`_.
