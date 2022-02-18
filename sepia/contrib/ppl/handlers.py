from copy import copy

# The base effect handler class.
class Handler:
    def __init__(self, fn):
        self.fn = fn

    # Handler stack.
    def stack(self):
        return self.fn.stack()

    # Effect handlers push themselves onto the handler stack.
    # Handlers later in the handler stack are applied first.
    def __enter__(self):
        # NOTE: Uncomment this to see how the handlers are stacked. 
        # print(f"pushing {self.__class__}")
        self.stack().append(self)

    def __exit__(self, *args, **kwargs):
        assert self.stack()[-1] is self
        self.stack().pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


# A first useful example of an effect handler. trace records the inputs and
# outputs of any primitive site it encloses, and returns a dictionary
# containing that data to the user.
class trace(Handler):
    def __enter__(self):
        super().__enter__()
        self.trace = dict()
        return self.trace

    # trace illustrates why we need postprocess_message in addition to
    # process_message: We only want to record a value after all other effects
    # have been applied.
    def postprocess_message(self, msg):
        assert (
            msg["type"] != "rv" or msg["name"] not in self.trace
        ), "sample sites must have unique names"
        self.trace[msg["name"]] = copy(msg)

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace


# A second example of an effect handler for setting the value at a sample site.
# This illustrates why effect handlers are a useful PPL implementation
# technique: We can compose trace and replay to replace values but preserve
# distributions, allowing us to compute the joint probability density of
# samples under a model. See the definition of elbo(...) below for an example
# of this pattern.
class condition(Handler):
    def __init__(self, fn, state):
        self.state = state
        super().__init__(fn)

    def process_message(self, msg):
        if msg["name"] in self.state.keys():
            msg["value"] = self.state[msg["name"]]

class biject(Handler):
    def process_message(self, msg):
        if msg["type"] == "rv" and not msg["observed"]:
            msg["unconstrained_value"] = msg["value"]
            msg["value"] = msg["fn"].to_constrained_space(
                msg["unconstrained_value"]
            )