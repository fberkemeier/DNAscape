Foundations
============

This section summarises the main quantities and variables used throughout
DNAscape to describe genome-wide DNA replication kinetics.

Models
---------------------

Functions
---------------------

Notation
---------------------

``firing_rate``
   Firing-rate profile for each origin of replication (ori) along the genome,
   representing the instantaneous probability per unit time that an ori
   initiates replication.

``efficiency``
   Efficiency of each ori, defined as the probability that an ori fires at
   least once during S phase.

``inter_ori_distances``
   Distances between neighbouring oris, represented either as an explicit
   list or as a probability distribution.

``num_oris``
   Total number of oris in the genome or within a specified genomic region.

``replication_timing``
   Genome-wide replication timing profile, giving the average replication
   time of each genomic locus.

``s_phase_duration``
   Total duration of S phase required for complete genome replication under
   the specified model or experimental conditions.

``fork_directionality``
   Replication fork directionality (RFD), describing the imbalance between
   leftward and rightward fork movement across the genome.

``fork_speeds``
   Replication fork speeds across the genome, represented either as a single
   scalar value or as a spatially varying array.

``num_active_forks``
   Number of replication forks actively progressing at a given time during
   S phase.

``total_num_forks``
   Total number of replication forks generated throughout the entire
   replication process.

``fraction_replicated``
   Fraction of replicated DNA as a function of time, ranging from zero at
   S-phase entry to one upon completion of genome replication.
