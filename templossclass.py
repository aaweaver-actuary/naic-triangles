class LossTriangleClassifier(nn.Module):
    def __init__(self,
                 input_shape,
                 num_classes=2,
                 num_conv_layers=4,
                 base_conv_nodes=64,
                 kernel_size=(2, 2),
                 stride=(1, 1),
                 padding=(1, 1),
                 linear_nodes=[128, 64],
                 linear_dropout=[0.1, 0.1],
                 relu_neg_slope=0.1,
                 activation=None,
                 output_activation=lambda x: F.softmax(x, dim=1)
                 ):
        super(LossTriangleClassifier, self).__init__()

        if activation is None:
            self.activation = lambda x: F.leaky_relu(x, relu_neg_slope)
        else:
            self.activation = activation

        self.output_activation = output_activation

        self.convolution_layers = nn.ModuleList()
        self.batch_normalization_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.linear_dropout_layers = nn.ModuleList()

        for i in range(num_conv_layers):
            node = base_conv_nodes * (2 ** i)
            self.convolution_layers.append(
                nn.Conv2d(1, node, kernel_size=kernel_size, stride=stride, padding=padding) if i == 0 else nn.Conv2d(base_conv_nodes * (2 ** (i - 1)), node, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            self.batch_normalization_layers.append(nn.BatchNorm2d(node))
            self.pooling_layers.append(nn.MaxPool2d(kernel_size[0], kernel_size[1]))

        self.flatten = nn.Flatten()

        for i, l in enumerate(linear_nodes):
            if i == 0:
                self.linear_layers.append(nn.Linear(self._get_flattened_size(input_shape), l))
            elif i == (len(linear_nodes) - 1):
                self.linear_layers.append(nn.Linear(linear_nodes[i - 1], num_classes))
            else:
                self.linear_layers.append(nn.Linear(linear_nodes[i - 1], l))

            self.linear_dropout_layers.append(nn.Dropout(linear_dropout[i]))

    def forward(self, x):
        for c, b, p in zip(self.convolution_layers, self.batch_normalization_layers, self.pooling_layers):
            x = c(x)
            x = b(x)
            x = self.activation(x)
            x = p(x)

        x = self.flatten(x)

        for l, d in zip(self.linear_layers, self.linear_dropout_layers):
            x = l(x)
            x = self.activation(x)
            x = d(x)

        return self.output_activation(x)

    def _get_flattened_size(self, input_shape):
        dummy_output = torch.zeros(1, *input_shape)
        for c, b, p in zip(self.convolution_layers, self.batch_normalization_layers, self.pooling_layers):
            dummy_output = c(dummy_output)
            dummy_output = b(dummy_output)
            dummy_output = self.activation(dummy_output)
            dummy_output = p(dummy_output)
        return dummy_output.numel()