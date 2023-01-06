# TorchFuzz

TorchFuzz is an open-source for fuzzing of pytorch models.

## Base code

The code is based on EvalDNN code.
You can find original code here: https://github.com/yqtianust/EvalDNN

## Usage

### Installation

Place torchfouzz folder and requirements.txt to your workspace folder.

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

Wrap model with PytorchModel().
use .runfuzzing() to start fuzzing.

dataset : instance of torch.utils.data.Dataset
    Dataset from which to load the data.
isTrain: boolean
    Check wheter dataset is train data or test data
isRandom: interger
    0 when want to check all parameters else positive integer
threshold: float
    Neuron coverage activate threshold
params_list: two-dimensional list or empty
    Empty if want to use base parameters else two-dimensional list of parameters

#### Base Parameters list
```python
params_list = [  # 테스트 파라미터
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