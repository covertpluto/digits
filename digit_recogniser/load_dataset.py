import pandas as pd
#import convert_data


def get_data():
    train_data = pd.read_csv("data/train1.csv")
    x_train = train_data.drop("label", axis=1) / 255.0
    y_train = train_data["label"]
    # print(x_train.shape)
    x_test = pd.read_csv("data/test1.csv") / 255.0
    return x_train, y_train, x_test
