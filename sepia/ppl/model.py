from copy import copy
from abc import ABC, abstractmethod
import numpy as np
from .handlers import *

def init_message(name, value, fn=None, args=None, kwargs=None, type="rv",
                 stop=False, given=None, observed=False):
    return dict(
        type = type,
        name = name,
        fn = fn,
        args = args,
        kwargs = kwargs,
        value = value,
        stop = stop,
        observed = observed,
        given=given
    )

class AbstractModel(ABC):
    def __init__(self, rng=np.random):
        self.rng = rng
        # Handlers later in the stack are first applied.
        self.handler_stack = []
        self.annotated = False

        # Parameter dependency graph.
        self.nodes = None
        self.parents = None
        self.children = None

    @abstractmethod
    def model(self, *args, **kwargs):
        """
        This is the only method that needs implementation.
        For example:

        import ppl
        class SimpleModel(ppl.AbstractModel):
            def model(self, y):
                self.annotated = True

                mu = self.rv("mu", Normal(0, 1))
                sigma = self.rv("sigma", Gamma(1, 1))
                self.rv("y", Normal(mu, sigma), given=["mu", "sigma])
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def stack(self):
        """Return handler_stack."""
        return self.handler_stack

    # apply_stack is called by ppl.sample and ppl.transform. It is responsible
    # for applying each Handler to each effectful operation.
    def apply_stack(self, msg):
        for pointer, handler in enumerate(reversed(self.stack())):
            handler.process_message(msg)
            # When a Handler sets the "stop" field of a message, it prevents
            # any Handlers below it on the stack from being applied.
            if msg["stop"]:
                break

        if msg["value"] is None:
            msg["value"] = msg["fn"](*msg["args"])

        # A Handler that sets msg["stop"] == True also prevents application
        # of postprocess_message by Handlers below it on the stack via the
        # pointer variable from the process_message loop.
        for handler in self.stack()[-pointer-1:]:
            handler.postprocess_message(msg)
        return msg

    # rv is an effectful version of Distribution.sample(...) When any effect
    # handlers are active, it constructs an initial message and calls
    # apply_stack.
    def rv(self, name, dist, *args, **kwargs):
        obs = kwargs.pop("obs", None)
        given = kwargs.pop("given", None)

        # if there are no active Handlers, we just draw a sample and return
        # it as expected:
        if not self.stack():
            return dist(*args, **kwargs)

        # Otherwise, we initialize a message...
        observed = not obs is None
        msg = init_message(name=name, fn=dist, args=args, kwargs=kwargs,
                           value=obs, given=given, observed=observed)

        # ...and use apply_stack to send it to the Handlers
        msg = self.apply_stack(msg)
        return msg["value"]

    def transform(self, name, value, *args, **kwargs):
        given = kwargs.pop("given", None)

        # if there are no active Handlers, we just draw a sample and return
        # it as expected:
        if not self.stack():
            return value

        # Otherwise, we initialize a message...
        msg = init_message(
            name=name,
            value=value,
            type="transform",
            given=given
        )

        # ...and use apply_stack to send it to the Handlers
        msg = self.apply_stack(msg)

        return msg["value"]

    # TODO: 
    # - Add capability to compute logpdf for a subset of the states?
    #       - This requires graph of children. 
    # - Add capability to compute loglikelihood only.
    #       - This requires graph of parents.
    def logpdf(self, state, *args, **kwargs):
        model_trace = trace(condition(self, state)).get_trace(*args, **kwargs)

        lp = 0
        for param in model_trace.values():
            if param["type"] == "rv":
                lp += np.sum(param["fn"].logpdf(param["value"]))

        return lp

    def prior_predictive(self, *args, **kwargs):
        """
        Generate a prior predictive draw from the model.

        - If `substate` (dict) is supplied, then the values will be fixed.
        - If `verbose` (bool) is supplied then the output detail can be
          controlled.
        """
        verbose = kwargs.pop("verbose", False)
        substate = kwargs.pop("substate", None)
        if substate is None:
            t = trace(self).get_trace(*args, **kwargs)
        else:
            t = trace(condition(self, substate)).get_trace(*args, **kwargs)

        if verbose:
            return t
        else:
            return {name: msg["value"] for name, msg in t.items()
                    if msg["type"] == "rv" and not msg["observed"]}

    def graph(self, *args, **kwargs):
        # TODO: 
        # author: alui
        # date: 11 Jan, 2022.
        # description: Can model annotations be automated?
        assert self.annotated, "Graph can only be traced if model is annotated."

        # Trace model.
        traced_model = trace(self).get_trace(*args, **kwargs)

        # inititalize the graph.
        self.nodes = set([])
        self.parents = dict()
        self.children = dict()
        for name in traced_model.keys():
            self.children[name] = set([])

        # First trace.
        for name, msg in traced_model.items():
            self.nodes.add(name)
            if msg["given"] is not None:
                self.parents[name] = msg["given"]
                for node in msg["given"]:
                    self.children[node].add(name)

        # On the second trace, connect nearest rvs via transforms.
        def next_rvs(parent, orig_parent=None):
            if orig_parent is None:
                orig_parent = parent
            children = copy(self.children[parent])
            for child in children:
                msg = traced_model[child]
                if msg["given"] is not None and parent in msg["given"]:
                    self.children[orig_parent].add(child)
                    if msg["type"] != "rv":
                        next_rvs(child, orig_parent=orig_parent)

        for name in traced_model:
            next_rvs(name)

        return dict(
            nodes=copy(self.nodes),
            parents=copy(self.parents),
            children=copy(self.children)
        )
