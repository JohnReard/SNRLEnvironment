from flax import linen as nn

class AgentNeuralNetwork(nn.Module):

    @nn.compact #removes need for setup, best for small models so maybe change later?
    def __call__(self, x):
        hiddenlayers : int
        outputs : int
        x = nn.Dense(features = self.hiddenlayers)(x)
        x = nn.Dense(features = self.outputs)(x)
        x = nn.tanh(x)
        return x
#to instantiate: (maybe put in training.py?)
model = AgentNeuralNetwork(hiddenlayers=100, outputs=4)

