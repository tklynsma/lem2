# LEM2 rule induction
## Description
A python2.7 implementation of LEM2 (Learning from Examples, Module 2): a rule induction algorithm based on rough set theory.

## Usage
```python
from lem2_classifier import LEM2Classifier

X_train, y_train, X_test, y_test = ...
lem2 = LEM2Classifier()
lem2.fit(X_train, y_train)
predictions = lem2.predict(X_test)
```

## Example
...

## Included example datasets
* Car Evaluation Dataset: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
* Lenses Dataset: https://archive.ics.uci.edu/ml/datasets/Lenses
* Play Tennis Dataset
* Soybean (Small) Dataset: https://archive.ics.uci.edu/ml/datasets/Soybean+(Small)
* Tic-Tac-Toe Endgame Dataset: https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
* Zoo Dataset: https://archive.ics.uci.edu/ml/datasets/Zoo