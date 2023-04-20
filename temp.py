here is my dataset class for preprocessing
class LossTriangleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        x = np.expand_dims(x, axis=0)  # Add channel dimension
        y = self.labels[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    @staticmethod
    def preprocess_data(dataframe):
        # Drop ay column
        dataframe = dataframe.drop('ay', axis=1)

        # Replace infinity values with NaN
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Remove nan values and normalize data
        scaler = StandardScaler()
        for triangle_id in dataframe['triangle_id'].unique():
            df_triangle = dataframe[dataframe['triangle_id'] == triangle_id].iloc[:, 2:]
            df_triangle = df_triangle.dropna(axis=1, how='all')
            dataframe.loc[dataframe['triangle_id'] == triangle_id, df_triangle.columns] = scaler.fit_transform(df_triangle)

        # Find the maximum number of rows
        max_rows = max([len(dataframe[dataframe['triangle_id'] == triangle_id]) for triangle_id in dataframe['triangle_id'].unique()])

        # Replace "nan" values with 0
        dataframe.fillna(0, inplace=True)


        # Create data and labels
        data = []
        labels = []
        for triangle_id in dataframe['triangle_id'].unique():
            df_triangle = dataframe[dataframe['triangle_id'] == triangle_id].iloc[:, 1:]
            label = df_triangle.iloc[-1, 0]
            triangle_data = df_triangle.iloc[:, 1:].values

            # Zero-padding
            nrows = triangle_data.shape[0]
            padded_triangle_data = np.zeros((max_rows, max_rows))
            for row_idx, row_data in enumerate(triangle_data):
                padded_triangle_data[row_idx, :max_rows-row_idx] = row_data[:max_rows-row_idx]

            data.append(padded_triangle_data)
            labels.append(label)

        # print(f"1. data[:, 0]: {data}")
        # print(f"1. data[:, 0]: {data[0, :, 0]}")

        # Convert data and labels to numpy arrays
        data = np.stack(data, axis=0)
        labels = np.array(labels)

        # print(f"2. data[0, :, 0]: {data[0, :, 0]}")

        # Check if there are any "nan" or infinite values in the data and labels
        assert not np.isnan(data).any(), "Data contains 'nan' values"
        assert not np.isinf(data).any(), "Data contains infinite values"
        assert not np.isnan(labels).any(), "Labels contain 'nan' values"
        assert not np.isinf(labels).any(), "Labels contain infinite values"

        return data, labels

    @staticmethod
    def create_datasets(dataframe, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        data, labels = LossTriangleDataset.preprocess_data(dataframe)
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_ratio, random_state=42)
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

        train_dataset = LossTriangleDataset(train_data, train_labels)
        val_dataset = LossTriangleDataset(val_data, val_labels)
        test_dataset = LossTriangleDataset(test_data, test_labels)

        return train_dataset, val_dataset, test_dataset


I am building a CNN model to classify a given triangle as either incremental (0) or cumulative (1). I have
reproduced my current processing class below. I am hitting a wall with accuracy and am thinking about 
adding additional features to the model. 

I want to adjust my class above to:
1. load the triangles.py file from my github repo: aaweaver-actuary/rocky3/rocky3/triangles.py
2. loop through the triangle_ids in the dataframe and create a Triangle object for each triangle_id
3. each triangle object will then run the .atu('vwa') method to calculate the vwa age-to-ult for each triangle
   - expect that vwa age-to-ult will be a strong predictor of incremental vs cumulative triangles (higher
     vwa age-to-ult = cumulative is more likely)
4. once this is calculated, each triangle will be processed as normal, but with the additional feature of the
   vwa age-to-ult, that gets normalized, zero padded, and added to the data separately from the triangle itself

Can you help me figure out what needs to be adjusted? I am not sure how to load the triangles.py file, or how
to do what I described.