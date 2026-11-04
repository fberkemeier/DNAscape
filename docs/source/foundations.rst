Core Concepts
=============

DNAscape connects measured genomic signals to a model of **origin firing**,
**fork movement** and **fork coalescence**. This chapter defines the quantities,
their units and the equations behind the main mappings. It also explains why
a good fit need not identify a unique replication programme.

For an application of timing-based inference to human genomes, see
[Berkemeier2025]_.

.. _concept-assays:

Experimental Profiles
---------------------

Repli-seq, SNS-seq and OK-seq measure different aspects of replication. Prepare
the quantity required by the next workflow:

* **Repli-seq:** a relative timing score needs an explicit time calibration
  before it represents minutes. For sixteen equally spaced chronological S-phase fractions,
  Data Manager also supports a weighted mean time or sigmoid T50 estimate
  using the specified S-phase duration.
* **SNS-seq and related origin-enrichment data:** preserve the spatial origin
  signal, then justify its scale before treating it as an initiation rate.
* **OK-seq:** preserve signed directionality and verify strand orientation.

See :ref:`datamanagement` for preparation controls and :ref:`quickstart`
for a specific Repli-seq example.

.. _concept-grids:

Coordinates And Dependence
--------------------------

A spatial profile contains one value per genomic bin. Regions use half-open
intervals: Start is included and End is excluded. For bin width
:math:`\Delta x`, a region of length :math:`L` needs approximately
:math:`L/\Delta x` values, with a partial final bin where necessary.

.. list-table:: Array conventions
   :header-rows: 1
   :widths: 22 22 56

   * - Dependence
     - Python shape
     - Meaning
   * - Constant
     - scalar
     - One value throughout the region and time domain.
   * - Space
     - ``(n,)``
     - One value per spatial bin.
   * - Time
     - ``(1, T)``
     - One value per time sample, shared across positions.
   * - Space-time
     - ``(n, T)``
     - Positions along rows, time samples along columns.

Genomic tracks use bp coordinates; Mapping and Simulation regions use kb.
Times are minutes. ``resolution_x`` denotes spatial bin width in kb, while
``resolution_t`` describes temporal input sampling. Recorded simulation
outputs use their returned ``time_grid``.

Changing resolution does not change the selected physical interval. For a
1 kb source, a 10 kb grid averages wider bins and has fewer values; a 0.1 kb
grid interpolates and has more values. Interpolation adds sampling locations,
not independent measurements. Plain profiles without coordinate metadata
need their source resolution to be supplied or interpreted explicitly.

.. _concept-units:

Initiation Rate And Fork Speed
------------------------------

Let :math:`I(x,t)` be potential initiation rate per unit **unreplicated** DNA
length and time, with units :math:`\mathrm{kb}^{-1}\mathrm{min}^{-1}`. A high
rate favours origin firing, but an origin can be passively replicated before
it fires. Actual origin efficiency is therefore a different quantity.

Let :math:`v(x,t)>0` be fork speed in kb/min. Symmetric forks move in opposite
directions at the same local speed. The corresponding grid speed is
:math:`v/\Delta x` in bins/min; the app performs this conversion internally.

The initiation-rate convention at each interface matters:

.. math::

   I_{\mathrm{bin}}(x,t) = I(x,t)\,\Delta x.

* The direct Python ``rmap`` interface uses the density :math:`I`.
* ``rsim`` and the app's initiation-rate inputs use
  :math:`I_{\mathrm{bin}}`, in initiations per bin per minute.
* App-to-app mappings and transfers handle their own boundary conversions.
  When passing a Python ``rmap`` result to ``rsim``, multiply by
  ``resolution_x``; see :ref:`pythonmodules`.

App region resampling averages or interpolates initiation-rate values under
its per-bin convention, rather than summing them. When constructing a grid
in Python to preserve a fixed physical density, recalculate
:math:`I_{\mathrm{bin}}` using the new bin width.

KJMA: Nucleation, Growth And Coalescence
----------------------------------------

The Kolmogorov-Johnson-Mehl-Avrami (KJMA) framework describes transformation by
random nucleation followed by growth and merging. In one-dimensional DNA,
origin firing nucleates a replicated domain, two forks expand it, and
opposing forks coalesce when domains meet. Potential initiations in DNA that
has already replicated do not create new origins. The one-dimensional theory
and its replication interpretation are developed in [Jun2005I]_ and
[Jun2005II]_.

For independent Poisson initiation attempts and prescribed fork movement,
consider all earlier origin firings that could reach position :math:`x` by
time :math:`t`. This set is its **backward replication cone**
:math:`\mathcal C(x,t)`. The position remains unreplicated if that cone
contains no potential initiation:

.. math::

   S(x,t) = \exp[-A(x,t)], \qquad
   A(x,t) = \iint_{\mathcal C(x,t)} I(y,\tau)\,dy\,d\tau,
   \qquad f(x,t) = 1-S(x,t).

Here :math:`f` is the local probability of replication, equivalently the
replicated fraction across an ensemble of identical models. A constant-speed
cone is triangular; this construction also supports inhomogeneous initiation
[Baker2014]_.

.. figure:: _static/replication_cone.svg
   :width: 660
   :alt: A constant-speed backward replication cone, showing potential origins that can reach a selected position and time.

   At constant speed, potential origin firings inside the shaded cone can
   reach the selected point. A finite region clips this cone; a periodic
   region wraps its spatial reach across the join.

Constant-speed example
~~~~~~~~~~~~~~~~~~~~~~

For constant speed :math:`v`, the points that can reach :math:`x` from time
:math:`\tau` lie between :math:`x-v(t-\tau)` and :math:`x+v(t-\tau)`.
On a sufficiently long homogeneous domain with constant initiation rate,
the cone integral becomes

.. math::

   A(t) = 2Iv\int_0^t (t-\tau)\,d\tau = Iv t^2,
   \qquad f(t)=1-\exp(-Iv t^2).

In this example the integrand uses constant :math:`I`; it is a rate multiplied
by the cone's growing spatial width. Boundaries, nonuniform inputs and
dependent initiation mechanisms require their appropriate model instead.

For a spatially uniform but time-dependent rate and speed, the same geometry
gives

.. math::

   R(t,\tau)=\int_\tau^t v(u)\,du, \qquad
   A(t)=2\int_0^t I(\tau)R(t,\tau)\,d\tau.

The distance travelled is the integral of speed over the intervening time,
not the final speed multiplied by elapsed time. When speed also depends on
position, fork paths satisfy :math:`dx/dt=\pm v(x,t)` and determine the
reachable region.

A time-dependent kinetic model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Time-dependent initiation and speed fits use a regularised Weibull-shaped
basis. Shape 1 gives a constant component, larger shapes allow an increasing
profile, and shapes below 1 allow a decreasing profile. The physical onset
interval keeps the first part finite independently of numerical resolution.
Fork speed also has a positive basal component. Spatial amplitudes and local
temporal shapes can vary across a space-time target.

The fitted temporal family is a modelling choice, rather than a universal
law of origin firing or independently sampled random fork travel times. The
amplitude carries the units of the kinetic field; the temporal basis is
dimensionless. See :ref:`kineticframework` for the exact parameterisation,
shared hazard equations, fitting procedure and identifiability limits.

Timing And Directionality
-------------------------

If the replication time at :math:`x` is :math:`T_x` and replication completes
with probability one, its mean is the integral of survival:

.. math::

   \overline T(x)=\mathbb E[T_x]=\int_0^\infty [1-f(x,t)]\,dt.

A finite calculation must capture the surviving tail. An explicit clock
origin or delay must also match the observation's time reference.
For the constant-rate, constant-speed infinite-domain example,

.. math::

   \overline T=\frac{\sqrt{\pi}}{2\sqrt{Iv}}.

This relation identifies the product :math:`Iv`, demonstrating why timing
alone cannot in general distinguish initiation from fork movement.

Fork directionality is the probability of rightward replication minus the
probability of leftward replication. With a symmetric speed that is constant
in time, its connection to the timing gradient is

.. math::

   D(x)=v(x)\frac{d\overline T(x)}{dx}, \qquad -1\le D(x)\le1.

This identity underlies timing/directionality mappings. Numerical derivatives
are sensitive to noise and edge treatment. Inferring speed is ill-conditioned
where timing is nearly flat; bounds and regularisation remain important.
A periodic derivative uses neighbours across the join rather than a forced
endpoint value.

.. _concept-events:

Fork Transport And Actual Event Densities
-----------------------------------------

Let :math:`\rho_+` and :math:`\rho_-` be right- and left-moving fork densities
per kb. In the symmetric mean-field model, replication progresses according
to [Gauthier2012]_

.. math::

   \partial_t f=v(\rho_++\rho_-).

Initiation creates one fork of each direction; coalescence removes one of
each. The corresponding transport equations are

.. math::

   \partial_t\rho_+ + \partial_x(v\rho_+) = j_i-j_c,
   \qquad
   \partial_t\rho_- - \partial_x(v\rho_-) = j_i-j_c,

with event-rate densities

.. math::

   j_i=I(1-f), \qquad
   j_c=\frac{2v\rho_+\rho_-}{1-f}.

The coalescence expression uses the mean-field closure for opposing forks;
it is not an assumption that arbitrary correlated stall or resource-limited
models obey the same equations. Numerically, the fully replicated limit
needs stable handling as :math:`1-f` approaches zero.

Both :math:`j_i` and :math:`j_c` have units **events/(kb min)**. Their time
integrals are **events/kb**; they are not normalised probability densities.
Total fork count is :math:`\int(\rho_++\rho_-)dx`.

In a completely replicated ensemble, directional probabilities weight fork
densities by the local speed:

.. math::

   P_\pm(x)=\int_0^\infty v(x,t)\rho_\pm(x,t)\,dt,
   \qquad D(x)=P_+(x)-P_-(x).

An unweighted ratio of integrated directional densities is not generally
appropriate when speed varies over time.

Simulation outputs count realised events in discrete spatial and temporal
bins, averaged across replicates. For comparison with a continuous rate
density, account for both bin width and recording interval. For example,
actual origin firing in a time interval is the integral of :math:`j_i` over
that interval and over the simulated region. A small spatial bin's origin
efficiency is approximately :math:`\Delta x\int j_i(x,t)dt`.

Fitting, Masks And Model Interpretation
---------------------------------------

A timing fit commonly minimises mean squared error on valid scored positions
:math:`\mathcal V`:

.. math::

   \operatorname{MSE}(\theta)=
   \frac{1}{|\mathcal V|}\sum_{x_j\in\mathcal V}
   [T_{\mathrm{model}}(x_j;\theta)-T_{\mathrm{observed}}(x_j)]^2.

The fitted parameters :math:`\theta` depend on the target family, supplied
inputs, bounds and region constraints. A numerical lower floor keeps a fitted
rate stable; it is distinct from an MSE stopping threshold. More iterations
optimise the chosen model and do not add experimental information.

A **mask** specifies which positions receive a regional operation. In
Mapping, transformed initial target values are held fixed there while other
values are fitted. In Data Manager and Simulation, the transform edits prepared
values or model inputs. In Analysis, a mask can restrict statistics without
compacting or shifting genomic indices.

Linear boundaries can omit origins outside the selected region. Periodic
models have a wrapped join but can still be sensitive to their assumed period.
Use matching coordinates and compare model predictions, residuals and
independent measurements. General :math:`I(x,t)` and :math:`v(x,t)` cannot
both be identified uniquely from one mean timing profile.

``rmap`` provides deterministic relationships and fitted predictions under
its route assumptions. ``rsim`` samples stochastic realisations, including
supported stalling and finite-factor mechanisms. Ensemble variation, finite
replicate counts and model mismatch can all contribute to differences between
a map and a simulated or measured profile.

.. _kineticframework:

Unified Mapping Framework
-------------------------

Mapping initiation rate, fork speed, replicated fraction and mean replication
timing uses one forward model. Inverse maps change the unknown kinetic field;
the equations predicting the observations remain the same. See
:ref:`mappingfunctions` for the app controls and :doc:`foundations` for units
and experimental interpretation.

One forward problem
~~~~~~~~~~~~~~~~~~~

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
[Baker2014]_.

Mean timing is the survival integral

.. math::

   \overline T(x)=\int_0^\infty e^{-H(x,t)}\,dt.

Thus predicting timing and fraction is one calculation with two observation
operators. A finite time grid must capture the survival tail. A small final
survival probability alone does not bound the remaining timing integral:
very weak continuing initiation can leave a slowly decaying tail. Check
convergence with a longer horizon or an independent tail bound in that case.
If a model
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
contributions matter [Gauthier2012]_.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
^^^^^^^^^^^^^^^^^^^^^^^^^

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
~~~~~~~~~~~~~~~~~~~~~~

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

For temporal speed with stationary initiation, adjacent source cells share
one primitive evaluation at their common edge. Their adjoint contributions
are combined at that edge before differentiation. General speed fits cache
characteristic positions and reconstruct their inverse-coordinate derivatives
from those positions, reducing repeated tracing and cache storage. Speed-only
adjoints sample the source density directly instead of building initiation
derivative records. These operations retain the same numerical model, bounds,
time grid and objective; a final prediction does not need an unused adjoint.
Space-time speed fits also sample the prescribed initiation primitive directly
when no source derivative record is needed, and evaluate Weibull fields with
broadcast temporal ages instead of copying masked space-time operands.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
[Baker2014]_.

Verification and benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Further Reading
---------------

.. [Berkemeier2025] F. Berkemeier, P. R. Cook and M. A. Boemo (2025).
   *DNA replication timing reveals genome-wide features of transcription and fragility.*
   Nature Communications 16, 4658.
   `Paper <https://doi.org/10.1038/s41467-025-59991-w>`_.

.. [Jun2005I] S. Jun, H. Zhang and J. Bechhoefer (2005).
   *Nucleation and growth in one dimension, part I: The generalized
   Kolmogorov-Johnson-Mehl-Avrami model.* Physical Review E 71, 011908.
   `Paper <https://doi.org/10.1103/PhysRevE.71.011908>`_.

.. [Jun2005II] S. Jun and J. Bechhoefer (2005).
   *Nucleation and growth in one dimension, part II: Application to DNA
   replication kinetics.* Physical Review E 71, 011909.
   `Paper <https://doi.org/10.1103/PhysRevE.71.011909>`_.

.. [Baker2014] A. Baker and J. Bechhoefer (2014).
   *Inferring the spatiotemporal DNA replication program from noisy data.*
   Physical Review E 89, 032703.
   `Paper <https://doi.org/10.1103/PhysRevE.89.032703>`_.

.. [Gauthier2012] M. G. Gauthier, P. Norio and J. Bechhoefer (2012).
   *Modeling Inhomogeneous DNA Replication Kinetics.* PLOS ONE 7, e32053.
   `Paper <https://doi.org/10.1371/journal.pone.0032053>`_.

.. [KineticGauthier2014] M. G. Gauthier, A. Dube and J. Bechhoefer (2014).
   *Numerical modeling of inhomogeneous DNA replication kinetics.*
   `Preprint <https://arxiv.org/abs/1408.5216>`_.
