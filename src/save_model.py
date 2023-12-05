import pickle
from src import config


def save(model, model_name: str):
    """
    Save a machine learning model using pickle.

    Parameters:
    - model: The machine learning model to be saved.
    - model_name (str): A descriptive name for the model.

    Prints:
    - Information about the saving process.
    """
    print("Saving with pickle")

    # Define the file path for saving the model
    filepath = config.DataConfig.models+model_name+".pkl"

    # Open the file in binary write mode and save the model using pickle
    with open(filepath, 'wb') as model_file:
        pickle.dump(model, model_file)

    # Print a message indicating the successful saving of the model
    print(f"NOTE: Model '{model_name}' saved at: {filepath}")

