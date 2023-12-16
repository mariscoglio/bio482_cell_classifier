# BIO-482 cell classification project  

This project aims to build a classifier to identify cell-classes in mouse barrel cortex given some information on the cells' activity, like their membrane potentials recorded through sweeps. In addition to some of the features provided in the project notebooks, additional features were computed and given to the different classifiers : 
- *ap_amp_mean* represents the average action potential amplitude for each sweep
- *ap_amp_cv* represents the coefficient of variation (= std / mean) of the action potential amplitudes for each sweep
- *mean_ap_upstroke* represents the average maximal rate of increase of the membrane potential during the action potential upstroke
- *ap_upstroke_cv* represents the coefficient of variation of the maximal rate of increase of the membrane potential during the action potential upstroke
- *mean_ap_downstroke* represents the average maximal rate of decrease of the membrane potential during the action potential downstroke
- *ap_downstroke_cv* represents the coefficient of variation of the maximal rate of decrease of the membrane potential during the action potential downstroke
- *isi_mean* represents the average duration between consecutive action potentials (inter-spike intervals)
- *isi_cv* represents the coefficient of variation of the duration between consecutive action potentials
- *irregularity_index* is an index indicating how regular the inter-spike intervals are. Concretely it is the mean of the difference between the durations of consecutive ISIs
- *adaptation_index* is another index indicating how regular the inter-spike intervals are. Concretely it is equal to the sum of the durations of consecutive ISIs divided by their difference. It is zero for a constant firing rate, negative for an increasing firing rate, and positive for a decreasing firing rate.
- *nb_bursts* is the number of bursts per sweep. Two consecutive action potentials are considered as being part of the same burst if the duration between them is less than 10ms. Bursts can contain more than two action potentials.
- *mean_burst_dur* is the mean duration of bursts 
- *burst_dur_cv* is the coefficient of variation of the duration of bursts 
