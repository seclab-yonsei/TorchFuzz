# TorchFuzz

**TorchFuzz** is an open-source for fuzzing of pytorch models.

## Base code

The code is based on EvalDNN code.  
You can find original code here: https://github.com/yqtianust/EvalDNN  
  
Mutation methods used are as follows. 
- translation
- scale
- shear
- contrast
- rotation
- brightness
- blur
- GaussianBlur
- MedianBlur
- bilateraFilter
  
You can find more information about mutation method in paper as follows.  
https://doi.org/10.1145/3293882.3330579

## Usage

### Installation

Place torchfuzz folder and requirements.txt to your workspace folder.

```
pip install -r requirements.txt
```

### Evaluate a model

```python
from torchfuzz.models.pytorch import PyTorchModel

measure_model = PyTorchModel(net)
measure_model.run_fuzzing(trainset, isTrain=True, threshold=0.5, isRandom=0)
measure_model.run_fuzzing(testset, isTrain=False, threshold=0.5, isRandom=0)
```

Wrap model with **PytorchModel()**.  
use **.runfuzzing()** to start fuzzing.

- **dataset** : instance of torch.utils.data.Dataset  
    Dataset from which to load the data.  
- **isTrain**: boolean  
    Check wheter dataset is train data or test data  
- **isRandom**: interger  
    0 when want to check all parameters else positive integer  
- **threshold**: float  
    Neuron coverage activate threshold  
- **params_list**: two-dimensional list or empty  
    Empty if want to use base parameters else two-dimensional list of parameters
#### Base Parameters list
```python
params_list = [
    [-20, -10, -5, 5, 10, 20],      # translation
    [5, 7, 12, 13, 15, 17],         # scale
    [-6, -5, -3, 3, 5, 6],          # shear
    [1, 2, 13, 20],                 # contrast
    [-60, -50, -40, 40, 50, 60],    # rotation
    [-90, -80, -70, 70, 80, 90],    # brightness
    [1, 2, 3, 5, 7, 9],             # blur
    [1, 3, 5, 7, 9, 11],            # GaussianBlur
    [1, 3, 5],                      # MedianBlur
    [6, 9]                          # bilateraFilter
    ]
```
#### Output
All datas are stored in **./cache** folder.

- **nc_arr.npy**: Coverage of train data  
- **corpus.pickle**: Mutationed data that are classified well  
- **crash_increase.pickle**: Mutationed data that are classified wrong and incrase coverage compared to train data  
- **crash_no_increase.pickle**: Mutationed data that are classified wrong and didn't incrase coverage compared to train data

All pickle files structure is as follows.
```python
[list of mutation data, list of label, list of original data, list of mutation parameter]
```
