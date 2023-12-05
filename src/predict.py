from src import setup_data
from src import config
from src import evaluate
import pickle
import pandas as pd


class LoadModel:
    def __init__(self, model_name: str, model_path: str, val: bool = True):
        """
        Initialize the LoadModel object.

        Parameters:
        - model_name (str): The name of the model.
        - model_path (str): The file path where the model is saved.
        - val (bool): Flag indicating whether to predict on validation data (True) or test data (False).
                      Default is True.
        """
        self.model_name = model_name
        self.model_path = model_path
        self.val = val

    def predict(self):
        """
        Load the model, make predictions, and perform actions based on the prediction.

        Raises:
        - ValueError: If the loaded model is None.
        """
        try:
            with open(self.model_path, 'rb') as model_file:
                model = pickle.load(model_file)
                print(model)

            if model is not None:
                if self.val:
                    prediction = model.predict(setup_data.load_data(config.DataConfig.X_val))
                    print(type(prediction))

                    # Classification report
                    evaluate.classificationreport(self.model_name, prediction)
                else:
                    prediction = model.predict(setup_data.load_data(config.DataConfig.test))

                    # Convert NumPy array to Pandas DataFrame
                    prediction = pd.DataFrame(prediction)

                    # Save prediction
                    prediction.to_csv(config.DataConfig.path + "/data/results/test_prediction.csv", index=False)
            else:
                raise ValueError("Loaded model is None.")

        except Exception as e:
            print(f"Error loading or predicting with the model: {e}")


def main():
    """
    Main function for user interaction and model prediction.
    """
    # Get user input for prediction type (validation or test)
    prompt = bool(int(input("Enter 1 (True) for prediction on validation data or or 0 (False) for test data: ")))

    print("\nSaved models available:\nlr: Logistic Reg\ndt: Decision Tree\
          \nrf: Random Forest\ngb: Gradient Boost\nxt: Extra Trees")

    # Dictionary of saved models
    saved_models = {'lr': config.DataConfig.lr, 'dt': config.DataConfig.dt,
                    'rf': config.DataConfig.rf, 'gb': config.DataConfig.gb,
                    'xt': config.DataConfig.xt}

    # Get user input for the saved model
    prompt2 = input("\nEnter saved model: ")
    for model in saved_models:
        if prompt2 == model:
            # Create LoadModel object and make predictions
            load = LoadModel(prompt2, saved_models[prompt2], prompt)
            load.predict()
        else:
            pass


if __name__ == "__main__":
    main()
