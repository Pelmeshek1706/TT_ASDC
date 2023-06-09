# TT_ASDC

Maded by Pelmeshek for Test Task

The Interface file contains code that must be executed as follows: 
<code>python interface.py C:\name_directory\with_test_images</code>

Untitked19.ipynb - project code with EDA and model training. 
To reduce the training time, the image resolution was reduced (to 256x256), the number of parameters is ~500k, also was added BatchNormalization. 
Parameters of model after one epoch: 
loss: 0.0040, dice_coef: 0.5176
val_loss: 1.7413e-06, val_dice_coef: 0.9885

