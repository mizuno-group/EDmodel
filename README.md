# EDmodel
Investigation of chemical structure recognition by encoder-decoder models in learning progress.  

## Publication
- [peer-reviewed publication](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00713-z)  
- [preprint](https://arxiv.org/abs/2210.16307)  

## Citation
Please cite the following if you use any contents of this repository.  
- DOI: 10.1186/s13321-023-00713-z  

## Authors
- [Shumpei Nemoto](https://github.com/Nemoto-S)  
    - main contributor  
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
    - correspondence  

## How to use
### Setup
```
git clone https://github.com/Nemoto-S/EDmodel.git
```
Please clone this repository locally and use it.

### Sample code
- descriptor generation
```python
import sys
sys.path.append("EDmodel")
from src.trainer import Generator
from src.models import Config

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
- 88nemo77[at]gmail.com  
- tadahaya[at]gmail.com  
    - lead contact  