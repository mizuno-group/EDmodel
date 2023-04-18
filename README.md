# EDmodel
Encoder-Decoder model for chemical structures.

## Authors
- [Shumpei Nemoto](https://github.com/Nemoto-S)  
    - main contributor  
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
    - correspondence  

## Note
This repository is under construction and will be officially released by [Mizuno group](https://github.com/mizuno-group).  

## Setup
```
git clone https://github.com/Nemoto-S/EDmodel.git
```
Please clone this repository locally and use it.

## Sample code

- descriptor generation
```python
import sys
sys.path.append("EDmodel")
from src.trainer import Generator
from src.model import Config

gen = Generator()
gen.load(Config(),"models/model_94.pth")
gen.get_descriptor(input_SMILES) # input_SMILES: List of your smiles
```

- structure generation
```python
gen.generate(descriptor) # descriptor: np.array or torch.tensor
```


## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- 88nemo77@gmail.com  
- tadahaya@gmail.com  
- tadahaya@mol.f.u-tokyo.ac.jp  