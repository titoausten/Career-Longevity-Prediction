from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier


class Models:
    def __init__(self, lr_params=None, dt_params=None, rf_params=None, gb_params=None, et_params=None):
        """
        Initialize the Models class.

        Parameters:
        - lr_params (dict): Hyperparameters for Logistic Regression (default: None).
        - dt_params (dict): Hyperparameters for Decision Tree Classifier (default: None).
        - rf_params (dict): Hyperparameters for Random Forest Classifier (default: None).
        - gb_params (dict): Hyperparameters for Gradient Boosting Classifier (default: None).
        - et_params (dict): Hyperparameters for Extra Trees Classifier (default: None).
        """
        self.lr_params = lr_params or {}
        self.dt_params = dt_params or {}
        self.rf_params = rf_params or {}
        self.gb_params = gb_params or {}
        self.et_params = et_params or {}

        self.lr = None
        self.dt = None
        self.rf = None
        self.gb = None
        self.et = None

    def logisticregression(self, new_params=None):
        """
        Create and return an instance of Logistic Regression model.

        Parameters:
        - new_params (dict): Override default hyperparameters if provided (default: None).

        Returns:
        - LogisticRegression: Instance of Logistic Regression model.
        """
        if new_params:
            self.lr_params = new_params
        self.lr = LogisticRegression(**self.lr_params)
        print("Logistic Regression model loaded!")
        return self.lr

    def decisiontreeclassifier(self, new_params=None):
        """
        Create and return an instance of Decision Tree Classifier model.

        Parameters:
        - new_params (dict): Override default hyperparameters if provided (default: None).

        Returns:
        - DecisionTreeClassifier: Instance of Decision Tree Classifier model.
        """
        if new_params:
            self.dt_params = new_params
        self.dt = DecisionTreeClassifier(**self.dt_params)
        print("Decision Tree Classifier model loaded!")
        return self.dt

    def randomforestclassifier(self, new_params=None):
        """
        Create and return an instance of Random Forest Classifier model.

        Parameters:
        - new_params (dict): Override default hyperparameters if provided (default: None).

        Returns:
        - RandomForestClassifier: Instance of Random Forest Classifier model.
        """
        if new_params:
            self.rf_params = new_params
        self.rf = RandomForestClassifier(**self.rf_params)
        print("Random Forest Classifier model loaded!")
        return self.rf

    def gradientboostingclassifier(self, new_params=None):
        """
        Create and return an instance of Gradient Boosting Classifier model.

        Parameters:
        - new_params (dict): Override default hyperparameters if provided (default: None).

        Returns:
        - GradientBoostingClassifier: Instance of Gradient Boosting Classifier model.
        """
        if new_params:
            self.gb_params = new_params
        self.gb = GradientBoostingClassifier(**self.gb_params)
        print("Gradient Boosting Classifier model loaded!")
        return self.gb

    def extratreesclassifier(self, new_params=None):
        """
        Create and return an instance of Extra Trees Classifier model.

        Parameters:
        - new_params (dict): Override default hyperparameters if provided (default: None).

        Returns:
        - ExtraTreesClassifier: Instance of Extra Trees Classifier model.
        """
        if new_params:
            self.et_params = new_params
        self.et = ExtraTreesClassifier(**self.et_params)
        print("Extra Trees Classifier model loaded!")
        return self.et
