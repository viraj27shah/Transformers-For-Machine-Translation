# Assignment 2
## Transformers

### Execution (To run this file)
```shell
# To run
1. python3 train.py

# To run test file and hyperparameter tuning
2. python3 test.py
```

### Files
- encoder.py : contains encoder block code
- decoder.py : contains decoder block code
- utils.py : contains all common blocks of transformer used in encoder and decoder
- build_transformer.py : all block are combined here of transformers
- train.py : training of model 
- test.py : evaluate on test set and hyperparmeter training
- testbleu.txt : bleu score with test set sentences
- transformer.pt : <a>https://drive.google.com/file/d/1JFCbrOqEHYWuCvtDH8lkvK92O-lc51_j/view?usp=sharing</a>
- report.pdf : analysis and hyperparameter tuning with question answers
- Readme.md : Instrunctions on how to run

### Asumptions 
- train.en,train,,fr,test.en,test.fr,dev.en,dev.fr input corpus must reside in same directory
- transformer.pt file must be present in the same folder. (uploaded in g drive)
