# Summer Project

## Preprocessing

IN PROGRESS !!!


## Model Design And Training

### Vanilla CNN Model


### Active Contraction State Classifier

The distribution of active and inactive contraction state is unequal.
Should we train another model serving for binary classification?

![active contraction state](src/readme_source/active_contraction_state_idea.png)

### Can auto-encoder assist us to solve the regression problem?

if classifier is helpful:
1. Semi-supervised Autoencoder (attach a binary classifier to enhance the sensitivity of active contraction frame)
2. Train two autoencoders after binary classifier

### Can other corresponding information (Angle/Angular Velocity) help  to solve the regression problem


### Can frame index help to solve the regression problem

----
***Next Step***

TODO

----
## Important Meeting Records

- April 1 2025
  - (Plot Problem) Professor.Yeo suggested me to label all axes with their meanings. It is helpful when doing presentations.
  - (Lingering Issue) Why you want to make a classifier? I need to prove for Professor.Yeo!
 

----
## Log


### FINISH [March 31 2025 (Morning)]

I need to determine whether some abnormal fluctuations in the signal are meaningful information or just noise.
The approach is to visualize the signal and analyze synchronized video recordings to observe muscle activity.
By comparing the signal changes with the muscle movements, I can assess whether the fluctuations are abnormal or irrelevant.

### FINISH [March 31 2025 (Night)]

I need to align all sample rates in mat file.

### FINISH [March 31 2025 (Night)]

Create Videos & Pre-processed dataset

### FINISH [March 31 2025 (Night)]

I find that the ultrasound image from different experiments are variant. Some of them are bright.
Some of them are dark. More specifically, the muscle texture may clear in some ultrasound images.
This is one of the main problem during data collections. I need to find some strategies to solve those
domain shift.

Now, I will normalize all images to mean=0, std=1. ***(Z-score Normalization / Standardization)***

I firstly calculate the global mean and global std among all the datasets, which I will use.

Then:

norm_ultrasound = (ultrasound - global_mean) / global_std

Comparison:

| Input Range                        | Loss Stability | Convergence Speed | Training Stability |
|-----------------------------------|----------------|-------------------|--------------------|
| `[0, 255]` (Raw Pixel Values)     | High Fluctuation | Slow            | Poor               |
| `[0, 1]` (Min-Max Normalization)  | Stable          | Fast              | Good               |
| `mean=0, std=1` (Z-score Standardization) | Very Stable     | Very Fast         | Excellent          |


### FINISH [April 1 2025 (Morning)]

Variant range of torque values!

Some datasets store torque values in really small values.
Some datasets store torque values in large values.

I guess the researcher standardized the torque value with its mean and stand deviation

May I get the exact mean and std to unify the value range reversely?

***Based on Professor.Yeo 's guidance, I understand I do not need to denoise the signal of angle & angular velocity, they are isometric.
I can directly calculate the average. Then, the original fluctuated signal will be replaced with average value.***

### FINISH [April 4 2025 (Afternoon)]

I need to implement codes to calculate average of angle and angular velocity.

As well as, some mat files has passive torque (showing as negative value). As Professor.Yeo reminded (Meeting [April 1 2025]).
I need to subtract the negative torque (average) for each sample index. [1 - (-1) = 1 + 1]

My work:

- Denoising
![denoise](src/readme_source/comparison_active_contraction_state_idea.png)
- Correcting (offset with passive torque)
![correction](src/readme_source/corrected_comparison_active_contraction_state_idea.png)


