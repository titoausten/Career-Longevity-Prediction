from sklearn.metrics import classification_report
from src import config
from src import setup_data
import pandas as pd


def classificationreport(model: str, prediction: pd.DataFrame = None):
    """
    Generate and print the classification report for a given model.

    Parameters:
    - model (str): The name or identifier of the model.
    - prediction (pd.DataFrame): The predicted values for the validation set.

    Returns:
    None
    """
    report = classification_report(setup_data.load_data(config.DataConfig.y_val), prediction)
    print(f"{model}: {report}")

    # Save the classification report to a file
    report_file_path = config.DataConfig.path+f"reports/{model}_classification_report.txt"
    with open(report_file_path, 'w') as report_file:
        report_file.write(report)
    print(f"NOTE: {model} classification report saved at: {report_file_path}")
