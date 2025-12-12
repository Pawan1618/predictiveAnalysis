
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold

class EnsembleModels:
    def __init__(self):
        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
        self.gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

    def train_rf_reg(self, X_train, y_train):
        self.rf_reg.fit(X_train, y_train)
        return self.rf_reg

    def train_ensemble_clf(self, X_train, y_train):
        models = {
            'RandomForest': self.rf_clf,
            'AdaBoost': self.ada_clf,
            'GradientBoosting': self.gb_clf
        }
        trained = {}
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            trained[name] = clf
        return trained

    def cross_validation(self, model, X, y, cv=5):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf)
        return scores.mean(), scores.std()
