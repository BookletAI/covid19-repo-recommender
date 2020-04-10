# Covid19 Repo Recommender

This Jupyter Notebook trains and deploys a model via MLFlow that recommends GitHub Covid19-related repos based on a programming language and keywords.

## Setup

```
python3 -m venv covid19-repo-recommender
source covid19-repo-recommender/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=covid19-repo-recommender
```

Then start the MLFlow server:

```
mlflow server
```
