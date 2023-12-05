import matplotlib.pyplot as plt
import seaborn as sns
from src import config


def heatmap(data):
    """
    Generate and save a heatmap showing the correlation between features.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing features.

    Returns:
    - None

    Saves a heatmap plot as a PNG file.
    """
    plt.figure(figsize=(16, 12))
    sns.heatmap(data.corr(), annot=True)
    plt.title("Heatmap showing the correlation between the features", fontsize=14)
    plt.savefig(config.DataConfig.figures+"heatmap.png")
