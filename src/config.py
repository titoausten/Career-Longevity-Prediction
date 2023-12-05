class DataConfig:
    """
    Configuration class for handling file paths and data-related settings.
    """
    path = "C:/Users/Tito Osadebey/Desktop/NBA-Career-Longevity-Prediction-/"
    train_path = path+"data/raw/Train_data.csv"
    test_path = path+"data/raw/Test_data.csv"
    target = "Target"
    column = "GP"

    # Interim data paths
    train = path+"data/interim/train.csv"
    train_x = path+"/data/interim/Xtrain.csv"
    val_x = path+"/data/interim/Xval.csv"
    feature = path+"/data/interim/feature.csv"
    label = path+"/data/interim/label.csv"
    
    # Processed data paths
    test = path+"data/processed/test.csv"
    X_train = path+"data/processed/Xtrain.csv"
    X_val = path+"data/processed/Xval.csv"
    y_train = path+"data/processed/ytrain.csv"
    y_val = path+"data/processed/yval.csv"

    # Saved models paths
    models = path+"models/"
    lr = models+"lr.pkl"
    dt = models+"dt.pkl"
    rf = models+"rf.pkl"
    gb = models+"gb.pkl"
    xt = models+"xt.pkl"

    # Figures and plots path
    figures = path+"reports/"


class ModelConfig:
    """
    Configuration class for handling model-related settings.
    """
    test_size = 0.2
    random_state = 42
