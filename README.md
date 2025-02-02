# Neural Network for Iris Classification

This repository contains a simple neural network implemented using PyTorch to classify the Iris dataset. The model consists of two hidden layers and uses ReLU activation.

## Features
- Implements a feedforward neural network with PyTorch
- Uses two hidden layers for classification
- Trains on the Iris dataset

## Installation
To set up the environment, install the required dependencies using pip:

```bash
pip install torch scikit-learn matplotlib
```

## Usage
Run the Jupyter Notebook to train and test the model:

```bash
jupyter notebook neural_network_iris_model.ipynb
```

## Model Architecture
```python
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
```

## Results
Below are sample results of the model's accuracy and loss during training.

![Model Training Accuracy](images/training_accuracy.png)
![Model Training Loss](images/training_loss.png)

## Contributing
Feel free to submit issues or pull requests if you'd like to improve the project.

## License
This project is licensed under the MIT License.

