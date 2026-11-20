from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CONFIG


def classifiers():
    seed = CONFIG["seed"]
    candidates = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1000, random_state=seed),
        "knn": KNeighborsClassifier(),
        "tree": DecisionTreeClassifier(random_state=seed),
        "forest": RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1),
        "svm": SVC(probability=True, random_state=seed),
    }
    return {name: make_pipeline(DictVectorizer(), StandardScaler(with_mean=False), model)
            for name, model in candidates.items()}
