import numpy as np

class Shaper:
    def __init__(self, state):
        self.shapes = dict()
        self.dim = self.vec(state).shape[0]

    def flatten(self, x):
        if np.isscalar(x):
            return np.array(x).flatten()
        else:
            return x.flatten()

    def vec(self, state):
        out = []
        self.shapes = dict()
        for name, value in state.items():
            self.shapes[name] = np.shape(value)
            out.append(self.flatten(value))
        return np.concatenate(out)

    def unvec(self, vec):
        state = dict()
        start = 0
        for name, shapes in self.shapes.items():
            num_elems = int(np.product(shapes))
            value = np.reshape(vec[start:start+num_elems], shapes)
            state[name] = value
            start += num_elems
        return state
