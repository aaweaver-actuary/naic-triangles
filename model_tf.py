@dataclass
class Model_tf:
  config : Optional[Any] = ModelConfig()
  X : Optional[Any] = np.array([[1,2,3],[4,5,6],[7,8,9]])
  y : Optional[Any] = np.array([1,1,1])
  
  name : Optional[str] = None
  model : Optional[Any] = None
  optimizer : Optional[Any] = None
  compiled : Optional[bool] = False
  accuracy_measures : Optional[dict] = None
  fit : Optional[Any] = None

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

  def fit(self, verbose=None):
    assert self.compiled, "Model has not been compiled yet."
    self.fitted_model = self.model.fit(
        self.X_train,
        self.y_train,
        batch_size=self.config.batch_size,
        epochs=self.config.epochs,
        verbose=verbose if verbose is not None else self.config.verbose,
        validation_data=(self.X_val, self.y_val))
    

  def plot(self, title=None, ax=None, label=None):
    if ax is None:
      fig, ax = plt.subplots(figsize=(15, 10))
  
    ax.plot(self.accuracy_measures[experiment],
            label=label,
            linewidth=2,
            alpha=0.7)
    
    if title is not None:
      plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()