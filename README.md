# BIO-482 cell classification project  

This project aims to build a classifier to identify cell-classes in mouse barrel cortex given some information on the cells' activity, like their membrane potentials recorded through sweeps. In addition to some of the features provided in the project notebooks, additional features were computed and given to the different classifiers : 
- *ap_amp_mean* represents the average action potential amplitude for each sweep; any missing values or NaNs were replaced with 0.
- *ap_amp_cv* represents the coefficient of variation (= std / mean) of the action potential amplitudes for each sweep; any missing values or NaNs were replaced with 0
- *mean_ap_upstroke* represents the average maximal rate of increase of the membrane potential during the action potential upstroke; any missing values or NaNs were replaced with 0
- *ap_upstroke_cv* represents the coefficient of variation of the maximal rate of increase of the membrane potential during the action potential upstroke; any missing values or NaNs were replaced with 0
- *mean_ap_downstroke* represents the average maximal rate of decrease of the membrane potential during the action potential downstroke; any missing values or NaNs were replaced with 0
- *ap_downstroke_cv* represents the coefficient of variation of the maximal rate of decrease of the membrane potential during the action potential downstroke; any missing values or NaNs were replaced with 0
- *isi_mean* represents the average duration between consecutive action potentials (inter-spike intervals); any missing values or NaNs were replaced with 50, as it made more sense to have it be a long time (50s) than a null time which would corresond to a situation where we have a lot of action potentials
- *isi_cv* represents the coefficient of variation of the duration between consecutive action potentials; any missing values or NaNs were replaced with 0
- *irregularity_index* is an index indicating how regular the inter-spike intervals are. Concretely it is the mean of the difference between the durations of consecutive ISIs; any missing values or NaNs were replaced with 0
- *adaptation_index* is another index indicating how regular the inter-spike intervals are. Concretely it is equal to the sum of the durations of consecutive ISIs divided by their difference. It is zero for a constant firing rate, negative for an increasing firing rate, and positive for a decreasing firing rate; any missing values or NaNs were replaced with 0, and very large values were capped to $10^{10}$
- *nb_bursts* is the number of bursts per sweep. Two consecutive action potentials are considered as being part of the same burst if the duration between them is less than 10ms. Bursts can contain more than two action potentials; any missing values or NaNs were replaced with 0
- *mean_burst_dur* is the mean duration of bursts; any missing values or NaNs were replaced with 0
- *burst_dur_cv* is the coefficient of variation of the duration of bursts; any missing values or NaNs were replaced with 0
