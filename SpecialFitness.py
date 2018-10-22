import sys

from collections import Sequence
from copy import deepcopy
from functools import partial
from operator import mul, truediv

class SpecialFitness(object):
    """The fitness is a measure of quality of a solution. If *values* are
    provided as a tuple, the fitness is initalized using those values,
    otherwise it is empty (or invalid).

    :param values: The initial values of the fitness as a tuple, optional.

    Fitnesses may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``,
    ``!=``. The comparison of those operators is made lexicographically.
    Maximization and minimization are taken care off by a multiplication
    between the :attr:`weights` and the fitness :attr:`values`. The comparison
    can be made between fitnesses of different size, if the fitnesses are
    equal until the extra elements, the longer fitness will be superior to the
    shorter.

    Different types of fitnesses are created in the :ref:`creating-types`
    tutorial.

    .. note::
       When comparing fitness values that are **minimized**,
       ``a > b`` will return :data:`True` if *a* is **smaller** than *b*
       and at least half of the following objectives are improved
       in this special case, the 4 objectives are: sum, srcError, diffSrcTar, diffCond
    """

    weights = None
    """The weights are used in the fitness comparison. They are shared among
    all fitnesses of the same type. When subclassing :class:`Fitness`, the
    weights must be defined as a tuple where each element is associated to an
    objective. A negative weight element corresponds to the minimization of
    the associated objective and positive weight to the maximization.

    .. note::
        If weights is not defined during subclassing, the following error will 
        occur at instantiation of a subclass fitness object: 

        ``TypeError: Can't instantiate abstract <class Fitness[...]> with
        abstract attribute weights.``
    """

    wvalues = ()
    """Contains the weighted values of the fitness, the multiplication with the
    weights is made when the values are set via the property :attr:`values`.
    Multiplication is made on setting of the values for efficiency.

    Generally it is unnecessary to manipulate wvalues as it is an internal
    attribute of the fitness used in the comparison operators.
    """

    def __init__(self, values=()):
        if self.weights is None:
            raise TypeError("Can't instantiate abstract %r with abstract "
                            "attribute weights." % (self.__class__))

        if not isinstance(self.weights, Sequence):
            raise TypeError("Attribute weights of %r must be a sequence."
                            % self.__class__)

        if len(values) > 0:
            self.values = values

    def getValues(self):
        return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        try:
            self.wvalues = tuple(map(mul, values, self.weights))
        except TypeError:
            _, _, traceback = sys.exc_info()
            raise TypeError, ("Both weights and assigned values must be a "
                              "sequence of numbers when assigning to values of "
                              "%r. Currently assigning value(s) %r of %r to a fitness with "
                              "weights %s."
                              % (self.__class__, values, type(values), self.weights)), traceback

    def delValues(self):
        self.wvalues = ()

    values = property(getValues, setValues, delValues,
                      ("Fitness values. Use directly ``individual.fitness.values = values`` "
                       "in order to set the fitness and ``del individual.fitness.values`` "
                       "in order to clear (invalidate) the fitness. The (unweighted) fitness "
                       "can be directly accessed via ``individual.fitness.values``."))

    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0

    def __hash__(self):
        return hash(self.wvalues)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        no_weights = len(self.wvalues)-1 #ignore the first as it is sum
        count = 0.0
        for index in range(1, no_weights+1):
            if self.wvalues[index] <= other.wvalues[index]:
                count = count+1.0
        se_ratio = count/no_weights
        return (self.wvalues[0] <= other.wvalues[0]) and (se_ratio >= 1.0)

    def __lt__(self, other):
        existSmaller = False
        for a,b in zip(self.wvalues, other.wvalues):
            if a > b:
                return False
            elif a < b:
                existSmaller = True
        return existSmaller
        #no_weights = len(self.wvalues)-1 #ignore the first as it is sum
        #count = 0.0
        #for index in range(1, no_weights+1):
        #    if self.wvalues[index] <= other.wvalues[index]:
        #        count = count+1.0
        #s_ratio = count/no_weights
        #return (self.wvalues[0] < other.wvalues[0]) and (s_ratio >= 1)

    def __eq__(self, other):
        return self.wvalues == other.wvalues

    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.

        It assumes that the elements in the :attr:`values` tuple are
        immutable and the fitness does not contain any other object
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__()
        copy_.wvalues = self.wvalues
        return copy_

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,
                              self.values if self.valid else tuple())
