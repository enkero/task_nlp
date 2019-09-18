##    Developed on:

conda 4.3.22 (Python 3.6)

numpy==1.16.4
pandas==0.24.2
scikit-learn==0.21.2
nltk==3.2.5
xgboost==0.6


##    Research and methodology testing in: 

nlp_task.ipynb


##    Run final model 

```sh
pytest run.py
```


##    Output files

predictions_prob.csv - probabilities for each label for test dataset

predictions_label.csv - class label for test dataset

