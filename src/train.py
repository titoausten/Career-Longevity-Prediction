from src import config
from src import setup_data
from src import model
from src import save_model


class Train:
    """
    Class for training machine learning models.

    Attributes:
    - features (pandas.DataFrame): The input features for training.
    - label (pandas.Series): The target labels for training.
    - model_instance (model.Models): An instance of the Models class for creating machine learning models.
    """
    def __init__(self, x_train, y_train):
        """
        Initializes a Train instance.

        Parameters:
        - x_train (pandas.DataFrame): The input features for training.
        - y_train (pandas.Series): The target labels for training.
        """
        self.features = x_train
        self.label = y_train
        self.model_instance = model.Models()

    def logreg(self):
        """
        Trains a Logistic Regression model.

        Returns:
        - sklearn.linear_model.LogisticRegression: Trained Logistic Regression model.
        """
        modelz = self.model_instance.logisticregression()
        modelz.fit(self.features, self.label)
        print("Training done!")
        return modelz

    def dectree(self):
        """
        Trains a Decision Tree model.

        Returns:
        - sklearn.tree.DecisionTreeClassifier: Trained Decision Tree model.
        """
        modelz = self.model_instance.decisiontreeclassifier()
        modelz.fit(self.features, self.label)
        print("Training done!")
        return modelz

    def randforest(self):
        """
        Trains a Random Forest model.

        Returns:
        - sklearn.ensemble.RandomForestClassifier: Trained Random Forest model.
        """
        modelz = self.model_instance.randomforestclassifier()
        modelz.fit(self.features, self.label)
        print("Training done!")
        return modelz

    def gradboost(self):
        """
        Trains a Gradient Boosting model.

        Returns:
        - sklearn.ensemble.GradientBoostingClassifier: Trained Gradient Boosting model.
        """
        modelz = self.model_instance.gradientboostingclassifier()
        modelz.fit(self.features, self.label)
        print("Training done!")
        return modelz

    def xtratrees(self):
        """
        Trains an Extra Trees model.

        Returns:
        - sklearn.ensemble.ExtraTreesClassifier: Trained Extra Trees model.
        """
        modelz = self.model_instance.extratreesclassifier()
        modelz.fit(self.features, self.label)
        print("Training done!")
        return modelz


def main():
    x_train = setup_data.load_data(config.DataConfig.X_train)
    y_train = setup_data.load_data(config.DataConfig.y_train)

    model_t = Train(x_train, y_train)

    print("\nList of available models:\nlr: Logistic Reg\ndt: Decision Tree\
          \nrf: Random Forest\ngb: Gradient Boost\nxt: Extra Trees")

    prompt = input("\nEnter initials for preferred training model: ")
    models = {'lr': model_t.logreg, 'dt': model_t.dectree,
              'rf': model_t.randforest, 'gb': model_t.gradboost,
              'xt': model_t.xtratrees}

    for mod in models:
        if prompt == mod:
            trained_model = models[prompt]()

            print("\nModel before saving:", trained_model)
            save_model.save(trained_model, prompt)
        else:
            pass


if __name__ == "__main__":
    main()
