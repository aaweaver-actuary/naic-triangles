# I have implemented the following set of classes to simplify model testing and hyperparameter tuning. 

# Can you please review the code and help me rewrite the classes and methods in such a way that:
#     1. The code is more efficient
#     2. The code can be used with either tensorflow OR pytorch
#     3. There are additional options for testing convolutional layers, etc.
   
# What do I need to do to make the model fit method work with both tensorflow and pytorch?

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from typing import Optional, List, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


def package_map(package_string : str):
    """
    A function to map the string values to the appropriate package. The
    default is 'tf' (i.e. TensorFlow). The other option is 'torch' (i.e.
    PyTorch).
    """
    out = {
      'tf': 'tf'
      , 'tensorflow': 'tf'
      , 'keras': 'tf'
      , 'torch': 'torch'
      , 'pytorch': 'torch'
      , 'pt': 'torch'
       }
    return out[package_string.lower()]

# map the string values to the appropriate package
def initialization_map(initialization_string : str = None, package : str = None):
    """
    A function to map the string values to the appropriate package. The default
    package is 'tf' (i.e. TensorFlow). The other option is 'torch' (i.e. PyTorch).



    """
    assert initialization_string is not None, "`initialization_string` cannot be None"

    # if no package is provided, default to 'tf'
    if package is None:
        package = 'tf'

    # map the string values to the appropriate package
    normal_map = {'tf': 'random_normal', 'torch': 'normal'}
    uniform_map = {'tf': 'random_uniform', 'torch': 'uniform'}
    zeros_map = {'tf': 'zeros', 'torch': 'zeros'}

    out = {'random_normal': normal_map,
           'normal': normal_map,
           'random_uniform': uniform_map,
           'uniform': uniform_map,
           'n': normal_map,
           'u': uniform_map,
           'zeros': zeros_map,
           'z': zeros_map
           }
    return out[initialization_string.lower()]

def activation_map(activation_string : str):
    relu_map = {'tf': 'relu', 'torch': 'ReLU'}
    sigmoid_map = {'tf': 'sigmoid', 'torch': 'Sigmoid'}
    softmax_map = {'tf': 'softmax', 'torch': 'Softmax'}
    tanh_map = {'tf': 'tanh', 'torch': 'Tanh'}
    leaky_relu_map = {'tf': 'leaky_relu', 'torch': 'LeakyReLU'}

    out = {
            'relu': relu_map,
            'sigmoid': sigmoid_map,
            'softmax': softmax_map,
            'tanh': tanh_map,
            'r': relu_map,
            's': sigmoid_map,
            'sm': softmax_map,
            't': tanh_map,
            'leaky_relu': leaky_relu_map,
            }
    return out[activation_string.lower()]

@dataclass
class ConvLayer:
    filters: Optional[int] = 32
    kernel_size: Optional[int] = 3
    activation: Optional[str] = 'relu'


@dataclass
class Hidden:
  nodes : Optional[List] = None
  activation : Optional[str]  = 'relu'

  def __post_init__(self):
    if self.nodes is None:
      self.nodes = [32, 64]

@dataclass
class Input:
  nodes : Optional[int] = None

@dataclass
class Output:
  nodes : Optional[int] = 3
  activation : Optional[str] = 'softmax'

@dataclass
class Initial:
  weights : Optional[str] = 'random_normal'
  bias : Optional[str] = 'zeros'

@dataclass
class ModelConfig:
  input : Optional[Input] = Input()
  output : Optional[Output] = Output()
  hidden : Optional[Hidden] = Hidden()
  initial : Optional[Initial] = Initial()
  normalization : Optional[str] = 'none'
  optimizer : Optional[str] = 'rmsprop'
  learning_rate : Optional[float] = 0.001
  regularizer : Optional[Any] = None
  dropout : Optional[float] = 0.0
  epochs : Optional[int] = 10
  batch_size : Optional[int] = 16
  train_split : Optional[float] = None
  validation_split : Optional[float] = None
  test_split : Optional[float] = None
  preprocessor : Optional[Any] = StandardScaler()
  verbose : Optional[bool] = 1
  loss_function : Optional[str] = 'categorical_crossentropy'
  metrics : Optional[List] = None
  conv: Optional[ConvLayer] = ConvLayer()

  def __post_init__(self):

    if self.metrics is None:
      
      self.metrics = ['accuracy']
    
    self.tvt = (self.train_split, self.validation_split, self.test_split)

    if ((self.train_split is None) &
        (self.validation_split is None) &
        (self.test_split is None)):
      self.train_split = 0.6
      self.validation_split=0.2
      self.test_split = 0.2
    else:
      x = ((1 if self.train_split is not None else 0) +
        (1 if self.validation_split is not None else 0) +
        (1 if self.test_split is not None else 0))
      errormsg = f"`train_split`: {self.train_split}\n"
      errormsg += f"`validation_split`: {self.validation_split}\n"
      errormsg += f"`test_split`: {self.test_split}\n"
      errormsg += "either specify all of (train, val, test) "
      errormsg += "or specify none of (train, val, test)"

      assert x==3, errormsg

class BaseModel(ABC):
    def __init__(self, config: ModelConfig, X: np.ndarray, y: np.ndarray, cv: bool = True, k: int = 3):
        self.config = config
        self.X = X
        self.y = y
        self.cv = cv
        self.k = k

        self.model = None
        self.optimizer = None
        self.compiled = False
        self.accuracy_measures = None
        self.fit = None
        self.fitted = None

        # Other attributes and method calls...

    # Other common methods

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def compile_model(self):
        pass

    @abstractmethod
    def fit_model(self, verbose: Optional[int] = None):
        pass

class TFModel(BaseModel):
    def __init__(self, config=None, X=None, y=None, cv=True, k=3, name=None):
        self.config = config if config else ModelConfig()
        self.X = X if X is not None else np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.y = y if y is not None else np.array([1, 1, 1])
        self.cv = cv
        self.k = k
        self.name = name
        self.model = None
        self.optimizer = None
        self.compiled = False
        self.accuracy_measures = None
        self.fit = None
        self.fitted = None

        if self.model is None:
            self.model = tf.keras.models.Sequential(name=self.name)

        if self.accuracy_measures is None:
            self.accuracy_measures = ['accuracy']

    def preprocess_data(self):
        # dummy vars for levels of the target
        self.y = pd.get_dummies(self.y)

        # split test set
        x1, self.X_test, y1, self.y_test = train_test_split(self.X,
                                                            self.y,
                                                            stratify=self.y,
                                                            test_size = self.config.test_split)
        
        # split train/val sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            x1,
            y1,
            stratify=y1,
            # split train/val from remaining data
            test_size = self.config.validation_split / (1 -self.config.test_split)
        )

        # add groups for cross validation if needed:
        temp = self.X_train.copy()
        self.cv_gp = cv_shuffle(temp, self.k)

        # fit standard scaler on X_train, apply to train, val, test
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

    def _add_layer(self, layer : int):
        self.model.add(keras.layers.Dense(self.config.hidden.nodes[layer],
                                          input_shape=(self.X.shape[1],),
                                          name=f"Dense-Layer-{layer}",
                                          kernel_initializer=self.config.initial.weights,
                                          bias_initializer=self.config.initial.bias,
                                          kernel_regularizer=self.config.regularizer,
                                          activation=self.config.hidden.activation))

    def _add_batch_normalization(self):
        if(self.config.normalization == 'batch'):
            self.model.add(keras.layers.BatchNormalization())

    def _add_dropout(self):
        if(self.config.dropout > 0.0):
            self.model.add(keras.layers.Dropout(self.config.dropout))

    def _get_optimizer(self):
        opts={
            'sgd': keras.optimizers.SGD(learning_rate=self.config.learning_rate)
            , 'rmsprop': keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
            , 'adam': keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            , 'adagrad': keras.optimizers.Adagrad(learning_rate=self.config.learning_rate)
        }
        self.optimizer = opts[self.config.optimizer]

    @abstractmethod
    def build_model(self):
        # loop through hidden nodes defined in config
        for layer in range(len(self.config.hidden.nodes)):
        
            # nothing to normalize/dropout for 1st hidden layer
            if layer==0: 
                self._add_layer(layer)
            
            # add batch normalization / dropout if indicated
            # to the hidden layers after the first
            else: 
                self._add_batch_normalization()
                self._add_dropout()
                self._add_layer(layer)

        # output layer
        self.model.add(keras.layers.Dense(
            self.config.output.nodes,
            name='Output-Layer',
            activation=self.config.output.activation
            )
        )

        # optimizer
        if self.optimizer is None:
            self._get_optimizer()

    @abstractmethod
    def compile_model(self):
        if self.optimizer is None:
            self.build_model()

        self.model.compile(
            loss=self.config.loss_function,
            optimizer=self.optimizer,
            metrics=self.config.metrics
        )
        self.compiled = True

    def summary(self):
        if self.compiled:
            self.model.summary()
        else:
            print('No model has been compiled. Run .compile_model() to compile.')

    # def cross_validation(self, k=3, verbose=0):
        # skip cross validation for now

    @abstractmethod
    def fit_model(self, verbose=None):
        assert self.compiled, "Model has not been compiled yet."
        self.fitted = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            verbose=verbose if verbose is not None else self.config.verbose,
            validation_data=(self.X_val, self.y_val))

class TorchModel(BaseModel):
    def __init__(self, config=None, X=None, y=None, cv=True, k=3, name=None):
        super().__init__(config, X, y, cv, k)
        self.name = name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1)
        }

    def preprocess_data(self):
        # dummy vars for levels of the target
        self.y = pd.get_dummies(self.y)

        # split test set
        x1, self.X_test, y1, self.y_test = train_test_split(self.X,
                                                            self.y,
                                                            stratify=self.y,
                                                            test_size=self.config.test_split)
        
        # split train/val sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            x1,
            y1,
            stratify=y1,
            test_size=self.config.validation_split / (1 - self.config.test_split)
        )

        # fit standard scaler on X_train, apply to train, val, test
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self.X_train = torch.tensor(self.scaler.transform(self.X_train), dtype=torch.float).to(self.device)
        self.X_val = torch.tensor(self.scaler.transform(self.X_val), dtype=torch.float).to(self.device)
        self.X_test = torch.tensor(self.scaler.transform(self.X_test), dtype=torch.float).to(self.device)

        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float).to(self.device)
        self.y_val = torch.tensor(self.y_val.values, dtype=torch.float).to(self.device)
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float).to(self.device)

    def build_model(self):
        class TorchNN(nn.Module):
            def __init__(self, config):
                super(TorchNN, self).__init__()
                layers = []
                input_dim = self.X_train.shape[1]
                for i, hidden_units in enumerate(self.config.hidden.nodes):
                    layers.append(nn.Linear(input_dim, hidden_units))
                    layers.append(self.activation_functions[self.config.hidden.activation])
                    if self.config.dropout > 0.0:
                        layers.append(nn.Dropout(self.config.dropout))
                    input_dim = hidden_units
                layers.append(nn.Linear(input_dim, self.config.output.nodes))
                layers.append(self.activation_functions[self.config.output.activation])
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        self.model = TorchNN(self.config).to(self.device)

    def compile_model(self):
        self.optimizer = {
            'sgd': optim.SGD(self.model.parameters(), lr=self.config.learning_rate),
            'rmsprop': optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate),
            'adam': optim.Adam(self.model.parameters(), lr=self.config.learning_rate),
            'adagrad': optim.Adagrad(self.model.parameters(), lr=self.config.learning_rate)
        }[self.config.optimizer]
        self.loss_function = nn.CrossEntropyLoss()
        self.compiled = True

    def fit_model(self, verbose: Optional[int] = None):
        assert self.compiled, "Model has not been compiled yet."
        train_dataset = TensorDataset(self.X_train, self.y_train.to(torch.long))
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for _, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % verbose == 0:
                print(f'Epoch {epoch + 1}/{self.config.epochs}, Loss: {epoch_loss / len(train_loader):.6f}')







@dataclass
class Model:
  config : Optional[Any] = ModelConfig()
  X : Optional[Any] = np.array([[1,2,3],[4,5,6],[7,8,9]])
  y : Optional[Any] = np.array([1,1,1])
  cv : Optional[bool] = True
  k : Optional[int] = 3
  
  name : Optional[str] = None
  model : Optional[Any] = None
  optimizer : Optional[Any] = None
  compiled : Optional[bool] = False
  accuracy_measures : Optional[dict] = None
  fit : Optional[Any] = None
  fitted : Optional[Any] = None

  def __post_init__(self):
    X_train, y_train = None, None
    X_val, y_val = None, None
    X_test, y_test = None, None

    if self.model is None:
      self.model = tf.keras.models.Sequential(name=self.name)

    if self.accuracy_measures is None:
      self.accuracy_measures = ['accuracy']

  def preprocess_data(self):
    # dummy vars for levels of the target
    self.y = pd.get_dummies(self.y)

    # split test set
    x1, self.X_test, y1, self.y_test = train_test_split(self.X,
                                                        self.y,
                                                        stratify=self.y,
                                                        test_size = self.config.test_split)
    
    # split train/val sets
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
        x1,
        y1,
        stratify=y1,
        # split train/val from remaining data
        test_size = self.config.validation_split / (1 -self.config.test_split)
    )

    # add groups for cross validation if needed:
    temp = self.X_train.copy()
    self.cv_gp = cv_shuffle(temp, self.k)

    # fit standard scaler on X_train, apply to train, val, test
    self.scaler = StandardScaler()
    self.scaler.fit(self.X_train)
    self.X_train = self.scaler.transform(self.X_train)
    self.X_val = self.scaler.transform(self.X_val)
    self.X_test = self.scaler.transform(self.X_test)

  def _add_layer(self, layer : int):
    self.model.add(
        keras.layers.Dense(
            self.config.hidden.nodes[layer],
            input_shape=(self.X.shape[1],),
            name=f"Dense-Layer-{layer}",
            kernel_initializer=self.config.initial.weights,
            bias_initializer=self.config.initial.bias,
            kernel_regularizer=self.config.regularizer,
            activation=self.config.hidden.activation))
    
  def _add_batch_normalization(self):
    if(self.config.normalization == 'batch'):
          self.model.add(keras.layers.BatchNormalization())

  def _add_dropout(self):
    if(self.config.dropout > 0.0):
          self.model.add(keras.layers.Dropout(self.config.dropout))

  def _get_optimizer(self):
    opts={
        'sgd': keras.optimizers.SGD(learning_rate=self.config.learning_rate)
        , 'rmsprop': keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        , 'adam': keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        , 'adagrad': keras.optimizers.Adagrad(learning_rate=self.config.learning_rate)
    }
    self.optimizer = opts[self.config.optimizer]


  def build_model(self):
    # loop through hidden nodes defined in config
    for layer in range(len(self.config.hidden.nodes)):
      
      # nothing to normalize/dropout for 1st hidden layer
      if layer==0: 
        self._add_layer(layer)
      
      # add batch normalization / dropout if indicated
      # to the hidden layers after the first
      else: 
        self._add_batch_normalization()
        self._add_dropout()
        self._add_layer(layer)

    # output layer
    self.model.add(keras.layers.Dense(
        self.config.output.nodes,
        name='Output-Layer',
        activation=self.config.output.activation
        )
    )

    # optimizer
    if self.optimizer is None:
      self._get_optimizer()

  def compile_model(self):
    if self.optimizer is None:
      self.build_model()
    self.model.compile(
        loss=self.config.loss_function,
        optimizer=self.optimizer,
        metrics=self.config.metrics
    )
    self.compiled = True

  def summary(self):
    if self.compiled:
      self.model.summary()
    else:
      print('No model has been compiled. Run .compile_model() to compile.')

  def cross_validation(self, k=3, verbose=0):
    assert self.compiled, "Model has not been compiled yet."
    self.cv_fit = {}
    self.k = k
    print("************ CROSS VALIDATION *********************************\n")
    for k0 in range(1, self.k+1):
      
      temp_name = f"{name}-fold-{k0}"
      
      print(f"Fitting {temp_name}")
      # self.cv_fit[temp_name] = {}
      # self.cv_fit['name'] = temp_name
      self.cv_fit[temp_name] = self.model.fit(
        pd.DataFrame(self.X_train).reset_index(drop=True).loc[pd.Series(self.cv_gp).reset_index(drop=True).ne(k0), :].to_numpy(),
        pd.DataFrame(self.y_train).reset_index(drop=True).loc[pd.Series(self.cv_gp).reset_index(drop=True).ne(k0), :].to_numpy(),
        batch_size=self.config.batch_size,
        epochs=self.config.epochs,
        verbose=verbose if verbose is not None else self.config.verbose,
        validation_data=(
            pd.DataFrame(self.X_train).reset_index(drop=True).loc[pd.Series(self.cv_gp).reset_index(drop=True).eq(k0), :].to_numpy(),
            pd.DataFrame(self.y_train).reset_index(drop=True).loc[pd.Series(self.cv_gp).reset_index(drop=True).eq(k0), :].to_numpy())
        )
    print("")
      
  def fit_model(self, verbose=None):
    assert self.compiled, "Model has not been compiled yet."
    self.fitted = self.model.fit(
        self.X_train,
        self.y_train,
        batch_size=self.config.batch_size,
        epochs=self.config.epochs,
        verbose=verbose if verbose is not None else self.config.verbose,
        validation_data=(self.X_val, self.y_val))
    
@dataclass
class HyperparameterTuning(Model):
    ### this is where the different tests, such as batch size, optimizer, etc. go

# What do I need to do to make the model fit method work with both tensorflow and pytorch?
# Please feel free to add classes, methods, and attributes as you see fit. Ask me questions if you need some clarification.


    batch_size_list : Optional[list] = None

    def __post_init__(self):
        if self.batch_size_list is None:
            self.batch_size_list = [4*i for i in range(int(128/4))]
    
    def test_batch_size(self, batch_size_list : list = None):
            assert self.compiled, "Model has not been compiled yet."
            batch_models = {}

            if batch_size_list is None:
                batch_size_list = [4*i for i in range(int(128/4))]

            # test batch sizes from 16 -> 128 in increments of 16
            # for batch_size in [64]:
            for batch_size in batch_size_list:
                # new model instance
                name = f"Batch-Size-{batch_size}"
                # print(f'name: {name}')
                batch_models[name] = Model(X=iris.drop(columns='species'),
                                            y=iris.species,
                                            name=name)

                # set batch size/epochs
                batch_models[name].config.batch_size = batch_size
                batch_models[name].config.epochs = 25
                batch_models[name].config.hidden.nodes = [32 for _ in range(3)]

                # process data/compile model
                batch_models[name].preprocess_data()
                batch_models[name].build_model()
                batch_models[name].compile_model()

                # model summary
                batch_models[name].summary()

                # fit model
                batch_models[name].fitted = batch_models[name].model.fit(
                        batch_models[name].X_train,
                        batch_models[name].y_train,
                        batch_size=batch_models[name].config.batch_size,
                        epochs=batch_models[name].config.epochs,
                        verbose=0,
                        validation_data=(batch_models[name].X_val, batch_models[name].y_val))
                
            # save results to class        
            self.batch_models = batch_models