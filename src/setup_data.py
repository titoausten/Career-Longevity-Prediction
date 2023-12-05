import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src import config
from src import visualize


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame.

    Parameters:
    - path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    data = pd.read_csv(path)
    return data


def round_to_int(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str) -> tuple:
    """
    Round specified column values to integers and save the updated DataFrames.

    Parameters:
    - train_df (pd.DataFrame): Training DataFrame.
    - test_df (pd.DataFrame): Test DataFrame.
    - column (str): Column to round to integers.

    Returns:
    - tuple[pd.DataFrame, pd.DataFrame]: Updated training and test DataFrames.
    """
    train_df[column] = train_df[column].apply(np.floor)
    test_df[column] = test_df[column].apply(np.floor)

    train_df.to_csv(config.DataConfig.path + "/data/interim/train.csv", index=False)
    test_df.to_csv(config.DataConfig.path + "/data/interim/test.csv", index=False)
    return train_df, test_df


def feature_separation(target: str, train_data: pd.DataFrame, loc: bool = True):
    """
    Separate features and labels from the training data and save them.

    Parameters:
    - target (str): Target column name.
    - train_data (pd.DataFrame): Training DataFrame.
    - loc (bool): Flag to include the location information.

    Returns:
    - tuple[pd.DataFrame, pd.Series]: Features and labels.
    """
    if loc:
        feature = train_data.drop([target], axis=1)
        label = train_data.loc[:, target]

        feature.to_csv(config.DataConfig.path + "/data/interim/feature.csv", index=False)
        label.to_csv(config.DataConfig.path + "/data/interim/label.csv", index=False)
        return feature, label
    else:
        feature = train_data.drop([target], axis=1)
        feature.to_csv(config.DataConfig.path + "/data/interim/feature.csv", index=False)
        return feature


def split(independent_vars: pd.DataFrame, dependent_var: pd.Series) -> tuple:
    """
    Split the data into training and validation sets and save them.

    Parameters:
    - independent_vars (pd.DataFrame): Independent variables (features).
    - dependent_var (pd.Series): Dependent variable (labels).

    Returns:
    - tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Split datasets.
    """
    x_train, x_val, y_train, y_val = train_test_split(independent_vars, dependent_var,
                                                      test_size=config.ModelConfig.test_size,
                                                      random_state=config.ModelConfig.random_state)

    x_train.to_csv(config.DataConfig.path + "/data/interim/Xtrain.csv", index=False)
    x_val.to_csv(config.DataConfig.path + "/data/interim/Xval.csv", index=False)
    y_train.to_csv(config.DataConfig.path + "/data/processed/ytrain.csv", index=False)
    y_val.to_csv(config.DataConfig.path + "/data/processed/yval.csv", index=False)
    return x_train, x_val, y_train, y_val


def scaler(x_train_data: pd.DataFrame, x_val_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """
    Scale the features using StandardScaler and save the scaled DataFrames.

    Parameters:
    - x_train_data (pd.DataFrame): Training features.
    - x_val_data (pd.DataFrame): Validation features.
    - test_data (pd.DataFrame): Test features.

    Returns:
    - tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Scaled datasets.
    """
    scaled = StandardScaler()
    x_scaled = scaled.fit_transform(x_train_data)
    x_val_scaled = scaled.transform(x_val_data)
    test_scaled = scaled.transform(test_data)

    x_train = pd.DataFrame(x_scaled)
    x_val = pd.DataFrame(x_val_scaled)
    tests = pd.DataFrame(test_scaled)

    x_train.to_csv(config.DataConfig.path + "/data/processed/Xtrain.csv", index=False)
    x_val.to_csv(config.DataConfig.path + "/data/processed/Xval.csv", index=False)
    tests.to_csv(config.DataConfig.path + "/data/processed/test.csv", index=False)

    return x_train, x_val, tests


if __name__ == "__main__":
    # Load data as Dataframe
    train = load_data(config.DataConfig.train_path)
    test = load_data(config.DataConfig.test_path)

    # Round float columns to int
    train, test = round_to_int(train, test, config.DataConfig.column)

    # Heatmap
    visualize.heatmap(train)

    # Feature Separation
    features, labelled = feature_separation(config.DataConfig.target, train)

    # Training and Validation data split
    split(independent_vars=features, dependent_var=labelled)

    # Feature Scaling
    scale = scaler(load_data(config.DataConfig.train_x), load_data(config.DataConfig.val_x), test)
