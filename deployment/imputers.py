from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

class CryoSleepImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to impute missing values in the 'CryoSleep' column based on the values
    in service-related columns. If a person has spent 0 on services, they are likely to have
    been in CryoSleep.

    Parameters:
    services_column : str
    The name of the service-related column
    used to impute the 'CryoSleep' variable.
    """
    def __init__(self, services_column):
        self.services_column = services_column

    def fit(self, X, y=None):
        """
        This method does not perform any fitting.
        Parameters:
        X : DataFrame
            The input data with the 'CryoSleep' and service-related columns.
        y : Optional[pd.Series]
            Ignored. Present for compatibility with the TransformerMixin interface.

        Returns:
        self : CryoSleepImputer
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Imputes the missing values in the 'CryoSleep' column based on the service-related column.
        If the service column value is 0, 'CryoSleep' is assumed 1 , otherwise 0.

        Parameters:
        X : DataFrame
            The input data with the 'CryoSleep' and service-related columns.

        Returns:
        DataFrame
            A DataFrame with the 'CryoSleep' column imputed and cast to integer type.
        """
        X = X.copy()

        missing_cryosleep_mask = X["CryoSleep"].isnull()
        X.loc[missing_cryosleep_mask, "CryoSleep"] = np.where(
            X.loc[missing_cryosleep_mask, self.services_column] == 0, 1, 0
        )

        return X[["CryoSleep"]].astype(int)

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names, in this case, the imputed 'CryoSleep' column.

        Parameters:
        input_features : Optional[List[str]]
            Ignored. Present for compatibility with the interface.

        Returns:
        List[str]
            A list containing the name of the output feature: 'CryoSleep'.
        """
        return ["CryoSleep"]


class HomePlanetImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for imputing missing values in the 'HomePlanet' column.
    The imputation logic is based on:
        - Deck values ('A', 'B', 'C' -> Europa, 'G' -> Earth, 'D' -> Mars),
        - Destination values (Destination 'PSOJ318.5-22' -> Earth),
        - Group and surname associations,
        - Filling remaining missing values with the most frequent 'HomePlanet' (mode).

    Attributes:
        mode : str
            The most frequent 'HomePlanet' in the data, used for filling remaining null values.
        """
    def __init__(self):
        self.mode = None

    def fit(self, X, y=None):
        """
        Fits the imputer by calculating the mode (most common value) of the 'HomePlanet' column.

        Parameters:
            X : DataFrame
                The input data containing the 'HomePlanet' column.
            y : Optional[pd.Series]
                Ignored. Present for compatibility with the TransformerMixin interface.

        Returns:
            self : HomePlanetImputer
                The fitted imputer.
        """
        self.mode = X['HomePlanet'].mode()[0]
        return self

    def transform(self, X):
        """
        Transforms the input data by imputing missing 'HomePlanet' values based on a series
        of rules and fills any remaining missing values with the mode.

        Parameters:
        X : DataFrame
            The input data with the 'HomePlanet', 'Deck', 'Destination', 'Group'
            and 'last_name' columns.

        Returns:
        DataFrame
            A DataFrame containing the imputed 'HomePlanet' column.
        """
        X = X.copy()
        X.loc[X['Deck'].isin(['A', 'B', 'C']), 'HomePlanet'] = 'Europa'
        X.loc[X['Deck'] == 'G', 'HomePlanet'] = 'Earth'
        X.loc[X['Deck'] == 'D', 'HomePlanet'] = 'Mars'

        X.loc[X['Destination'] == 'PSOJ318.5-22', 'HomePlanet'] = 'Earth'

        # Fill based on Group (Assuming group is stored in a 'Group' column)
        for group in X['Group'].unique():
            group_data = X[X['Group'] == group]
            if group_data['HomePlanet'].isnull().any() and len(group_data) > 0:
                most_common_planet = group_data['HomePlanet'].mode()
                if len(most_common_planet) != 0:
                    X.loc[X['Group'] == group, 'HomePlanet'] = most_common_planet[0]

        # Fill based on Surname
        for surname in X['last_name'].unique():
            surname_data = X[X['last_name'] == surname]
            if surname_data['HomePlanet'].isnull().any() and len(surname_data) > 0:
                most_common_planet = surname_data['HomePlanet'].mode()
                if len(most_common_planet) != 0:
                    X.loc[X['last_name'] == surname, 'HomePlanet'] = most_common_planet[0]

        # Fill remaining null values with mode
        X.fillna({"HomePlanet": self.mode}, inplace=True)
        return X[["HomePlanet"]]

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names, in this case, the imputed 'HomePlanet' column.

        Parameters:
        input_features : Optional[List[str]]
            Ignored. Present for compatibility with the interface.

        Returns:
        List[str]
            A list containing the name of the output feature.
        """
        return ["HomePlanet"]


class ServiceImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for imputing missing values in service-related columns
    based on the 'CryoSleep' status.
    If a person is in CryoSleep (CryoSleep == 1), any missing service values are replaced with 0.

    Parameters:
    services : List[str]
        A list of service-related columns in the dataset that need to be imputed.
    """
    def __init__(self, services):
        self.services = services

    def fit(self, X, y=None):
        """
       This method does not perform any fitting.

       Parameters:
       X : DataFrame
           The input data containing the service columns and 'CryoSleep' column.
       y : Optional[pd.Series]
           Ignored. Present for compatibility with the TransformerMixin interface.

       Returns:
       self : ServiceImputer
           Returns the instance itself.
       """
        return self

    def transform(self, X):
        """
        Transforms the input data by replacing missing values in the service columns with 0
        where 'CryoSleep' equals 1.

        Parameters:
        X : DataFrame
            The input data containing the service columns and the 'CryoSleep' column.

        Returns:
        DataFrame
            A DataFrame with the imputed service columns.
        """
        X = X.copy()

        for service in self.services:
            # Replace NaNs with 0 where CryoSleep == 1
            X[service] = np.where(
                (X["CryoSleep"] == 1) & (X[service].isna()),
                0,
                X[service]
            )
        return X[self.services]

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names, in this case, the imputed services columns.

        Parameters:
        input_features : Optional[List[str]]
            Ignored. Present for compatibility with the interface.

        Returns:
        List[str]
            A list containing the name of the output features.
        """
        return self.services


class LinearRegressionImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that imputes missing values in a target column using
    a linear regression model based on a specified feature column.
    The model is trained on the available data, and predictions are made for the missing values.

    Parameters:
    target_col : str
        The name of the column with missing values to be imputed.
    feature_col : str
        The name of the column to use as a feature for the linear regression model.
    """
    def __init__(self, target_col, feature_col):
        self.target_col = target_col
        self.feature_col = feature_col

    def fit(self, X, y=None):
        """
        Fits the linear regression model using the data available for the target column.

        Parameters:
        -----------
        X : DataFrame
            The input data containing both the target column and the feature column.
        y : Optional[pd.Series]
            Ignored. Present for compatibility with the TransformerMixin interface.

        Returns:
        --------
        self : LinearRegressionImputer
            The fitted imputer.
        """
        data_with_values = X[X[self.target_col].notna()]
        X_train = data_with_values[[self.feature_col]]
        y_train = data_with_values[self.target_col]

        # Fit the model
        self.regressor = LinearRegression()
        self.regressor.fit(X_train, y_train)
        return self

    def transform(self, X):
        """
        Transforms the input data by predicting missing values in the
        target column using the fitted model.

        Parameters:
        -----------
        X : DataFrame
            The input data containing the feature column and
             potentially missing values in the target column.

        Returns:
        --------
        DataFrame
            A DataFrame containing the imputed values for the target column.
        """
        X = X.copy()

        X_missing = X[X[self.target_col].isna()]

        # return the data if there are no missing values
        if X_missing.shape[0] == 0:
            return X[[self.target_col]]

        X_train = X_missing[[self.feature_col]]
        predicted = self.regressor.predict(X_train)

        predicted = np.clip(predicted, 0, None)

        X.loc[X[self.target_col].isna(), self.target_col] = predicted

        return X[[self.target_col]]

    def get_feature_names_out(self, input_features=None):
        """
        Returns the name of the target column after transformation.

        Parameters:
        input_features : Optional[List[str]]
            Ignored. Present for compatibility with the interface.

        Returns:
        List[str]
            A list containing the name of the target column.
        """
        return [self.target_col]


class CabinGroupTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that categorizes cabin numbers into discrete groups
    based on specified bins.

    Parameters:
    bins : List[float]
        The boundaries for the bins to categorize the cabin numbers.
    labels : List[str]
        The labels for the bins corresponding to each range defined by `bins`.
    """
    def __init__(self, bins, labels):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        """
        Fits the transformer. In this case, it does not learn anything
        and simply returns itself.

        Parameters:
        X : DataFrame
            The input data containing the 'CabinNumber' column.
        y : Optional[pd.Series]
            Ignored. Present for compatibility with the TransformerMixin interface.

        Returns:
        self : CabinGroupTransformer
            The fitted transformer instance.
        """
        return self

    def transform(self, X):
        """
       Transforms the input data by binning 'CabinNumber' into
       discrete groups based on the specified bins and labels.

       Parameters:
       X : DataFrame
           The input data containing the 'CabinNumber' column to be transformed.

       Returns:
       DataFrame
           A DataFrame containing a single column 'CabinGroup' with the
           binned categories for the corresponding cabin numbers,
           converted to integer type.
       """
        X = X.copy()
        X["CabinGroup"] = pd.cut(X["CabinNumber"],
                                 bins=self.bins,
                                 include_lowest=True,
                                 labels=self.labels)
        return X[["CabinGroup"]].astype(int)

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names, in this case, the created CabinGroup.

        Parameters:
        input_features : Optional[List[str]]
            Ignored. Present for compatibility with the interface.

        Returns:
        List[str]
            A list containing the name of the output feature.
        """
        return ["CabinGroup"]