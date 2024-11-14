import pandas as pd
import numpy as np

def create_new_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
        Creates new features for the given dataset.
        Args:
            dataset (pd.DataFrame): A DataFrame containing at least the following columns:
                - 'Name' (str): The full name of the passenger.
                - 'CryoSleep' (str/bool): Whether the passenger is in cryosleep.
                - 'PassengerId' (str): The passenger's ID.
                - 'Age' (float): The passenger's age.
                - 'RoomService', 'Spa', 'VRDeck' (float): Numeric features representing
                  amounts spent on various services.
        Returns:
            pd.DataFrame: The modified DataFrame with new features:
                - 'last_name' (str or NaN): The extracted last name from the 'Name' column.
                - 'CryoSleep' (bool): Converted to a boolean column.
                - 'Group' (int): Group ID extracted from the first 4 characters of 'PassengerId'.
                - 'child' (int): A binary column indicating whether the passenger is a child (Age < 18).
                - 'RoomSpaVrDeck' (float): Sum of 'RoomService', 'Spa', and 'VRDeck' columns.
        """
    dataset['last_name'] = (dataset['Name'].apply(
        lambda x: x.split(' ')[-1] if len(x.split(' ')) > 1 else np.nan))
    dataset["CryoSleep"] = dataset["CryoSleep"].astype(bool)
    dataset.loc[:, "Group"] = dataset["PassengerId"].str[:4].astype(int)
    dataset["child"] = dataset["Age"].apply(lambda x: 1 if x < 18 else 0)
    dataset["RoomSpaVrDeck"] = dataset[["RoomService", "Spa", "VRDeck"]].sum(axis=1)
    return dataset