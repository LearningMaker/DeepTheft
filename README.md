# DeepTheft
### ABOUT
This is the training source code for the offline phase of DeepTheft, which comprises two steps.

You can access the complete RAPL-based dataset at https://drive.google.com/drive/folders/1MEuaUYt_js9meAMut9WIERhtnlIn_hqK?usp=drive_link. The dataset includes "data.h5" and "hp.h5," representing the energy trace and hyperparameters, respectively. To execute the code successfully, please ensure that you place these files in the "datasets" folder within the root directory.

### DEPENDENCIES
Our code is implemented and tested on PyTorch. Following packages are used by our code.
- `troch==1.10.1`
- `torchvision==0.11.2`
- `numpy==1.22.0`
- `python-Levenshtein==0.12.2`

### RUN
Step 1 of the offline training phase.
```python
python Step1_Network_Structure_Recovery/train.py
```
Step 2 of the offline training phase.
```python
python Step2_Layer-wise_Hyperparameter_Inferring/train.py
```
