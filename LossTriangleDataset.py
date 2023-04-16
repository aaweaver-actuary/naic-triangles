import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LossTriangleDataset(Dataset):
    """
    LossTriangleDataset class

    This class is used to create a dataset from a dataframe containing the loss triangle data, 
    and to preprocess the data before passing it to the model.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be used for training, validation or testing.
    labels : numpy.ndarray
        The labels to be used for training, validation or testing.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(index)
        Returns the data and label at the given index.
    preprocess_data(dataframe)
        Preprocesses the data from the dataframe.
    create_datasets(dataframe, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        Creates the train, validation and test datasets from the dataframe.
    """
    def __init__(self
                 , data : np.ndarray
                 , labels : np.ndarray
                 ) -> None:
        """
        Initializes the LossTriangleDataset class.
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self
                    , index : int
                    ) -> tuple:
        """
        Returns the data and label at the given index.

        Parameters
        ----------
        index : int
            The index of the data and label to be returned.

        Returns
        -------
        tuple
            The data and label at the given index.
        """
        
        x = self.data[index]
        x = np.expand_dims(x, axis=0)  # Add channel dimension
        y = self.labels[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    @staticmethod
    def preprocess_data(dataframe: pd.DataFrame) -> tuple:
        """
        Preprocesses the data from the dataframe.

        Preprocessing steps:
        1. Drop the ay column.
        2. Replace infinity values with NaN.
        3. Remove nan values and normalize data.
        4. Find the maximum number of rows.
        5. Replace "nan" values with 0.
        6. Create data and labels.
        7. Convert data and labels to numpy arrays.
        8. Check if there are any "nan" or infinite values in the data and labels.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the loss triangle data.
        """
        # Drop ay column
        dataframe = dataframe.drop('ay', axis=1)

        # Replace infinity values with NaN
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Remove nan values and normalize data
        scaler = StandardScaler()
        for triangle_id in dataframe['triangle_id'].unique():

            # pick out a single triangle and keep only the columns with data
            df_triangle = dataframe[dataframe['triangle_id'] == triangle_id].iloc[:, 2:]
            
            # remove columns with all nan values
            df_triangle = df_triangle.dropna(axis=1, how='all')
            
            # normalize the original data
            dataframe.loc[dataframe['triangle_id'] == triangle_id, df_triangle.columns] = scaler.fit_transform(df_triangle)

        # Find the maximum number of rows
        max_rows = max([len(dataframe[dataframe['triangle_id'] == triangle_id]) for triangle_id in dataframe['triangle_id'].unique()])

        # Replace "nan" values with 0
        dataframe.fillna(0, inplace=True)


        # Create data and labels
        data = []
        labels = []

        # loop over all triangles, and create a list of data and labels
        for triangle_id in dataframe['triangle_id'].unique():
            df_triangle = dataframe[dataframe['triangle_id'] == triangle_id].iloc[:, 1:]
            label = df_triangle.iloc[-1, 0]
            triangle_data = df_triangle.iloc[:, 1:].values

            # Zero-padding
            padded_triangle_data = np.zeros((max_rows, max_rows))
            for row_idx, row_data in enumerate(triangle_data):
                padded_triangle_data[row_idx, :max_rows-row_idx] = row_data[:max_rows-row_idx]

            data.append(padded_triangle_data)
            labels.append(label)

        # Convert data and labels to numpy arrays
        data = np.stack(data, axis=0)
        labels = np.array(labels)

        # Check if there are any "nan" or infinite values in the data and labels
        assert not np.isnan(data).any(), "Data contains 'nan' values"
        assert not np.isinf(data).any(), "Data contains infinite values"
        assert not np.isnan(labels).any(), "Labels contain 'nan' values"
        assert not np.isinf(labels).any(), "Labels contain infinite values"

        return data, labels

    @staticmethod
    def create_datasets(dataframe : pd.DataFrame
                        , train_ratio : float = 0.6
                        , val_ratio : float = 0.2
                        , test_ratio : float = 0.2
                        ) -> tuple:
        """
        Creates the train, validation and test datasets from the dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the loss triangle data.
        train_ratio : float, optional
            The ratio of the data to be used for training, by default 0.6
        val_ratio : float, optional
            The ratio of the data to be used for validation, by default 0.2
        test_ratio : float, optional
            The ratio of the data to be used for testing, by default 0.2

        Returns
        -------
        tuple
            The train, validation and test datasets.
        """
        # Preprocess data
        data, labels = LossTriangleDataset.preprocess_data(dataframe)

        # Split data into train and test
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_ratio, random_state=42)
        
        # Split `train` data into train and validation
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

        # Create datasets from the data and labels
        train_dataset = LossTriangleDataset(train_data, train_labels)
        val_dataset = LossTriangleDataset(val_data, val_labels)
        test_dataset = LossTriangleDataset(test_data, test_labels)

        return train_dataset, val_dataset, test_dataset
