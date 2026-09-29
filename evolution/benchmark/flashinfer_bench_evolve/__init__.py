"""FlashInfer kernel benchmark harness and task suite.

A regular (non-namespace) package on purpose: an installed copy must not be
shadowed or merged with same-named directories elsewhere on ``sys.path``,
and a source checkout must win over an installed copy when running from the
checkout.
"""

__version__ = "0.1.0"
