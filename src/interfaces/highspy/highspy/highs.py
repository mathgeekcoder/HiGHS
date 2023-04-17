from .highs_bindings import (
    ObjSense,
    MatrixFormat,
    HessianFormat,
    SolutionStatus,
    BasisValidity,
    HighsModelStatus,
    HighsBasisStatus,
    HighsVarType,
    HighsStatus,
    HighsLogType,
    CallbackTuple,
    HighsSparseMatrix,
    HighsLp,
    HighsHessian,
    HighsModel,
    HighsSolution,
    HighsBasis,
    HighsInfo,
    HighsOptions,
    _Highs,
    kHighsInf,
    HIGHS_VERSION_MAJOR,
    HIGHS_VERSION_MINOR,
    HIGHS_VERSION_PATCH,
)

from itertools import groupby
from operator import itemgetter
from decimal import Decimal

class Highs(_Highs):
    """HiGHS solver interface"""
    __slots__ = ['_batch', '_vars', '_cons', '_log_callback_tuple']

    def __init__(self):
        super().__init__()
        self._log_callback_tuple = CallbackTuple()
        
        self._batch = highs_batch(self)
        self._vars = []
        self._cons = []


    def setLogCallback(self, func, callback_data):
        self._log_callback_tuple.callback = func
        self._log_callback_tuple.callback_data = callback_data
        super().setLogCallback(self._log_callback_tuple)

    # reset the objective and sense, then solve
    def minimize(self, obj):
        # if we have a single variable, wrap it in a linear expression
        if isinstance(obj, highs_var) == True:
            obj = highs_linear_expression(obj)

        if isinstance(obj, highs_linear_expression) == False or obj.sense != None:
            raise Exception('Objective cannot be an inequality') 

        self.update()
        super().changeObjectiveSense(ObjSense.kMinimize)
        
        # reset objective
        super().changeColsCost(self.numVars, range(self.numVars), [0]*self.numVars)

        # if we have duplicate variables, add the vals
        vars,vals = zip(*[(var, sum(v[1] for v in Vals)) for var, Vals in groupby(sorted(zip(obj.vars, obj.vals)), key=itemgetter(0))])
        super().changeColsCost(len(vars), vars, vals)
        super().changeObjectiveOffset(-obj.RHS)

        return super().run()

    # reset the objective and sense, then solve
    def maximize(self, obj):
        # if we have a single variable, wrap it in a linear expression
        if isinstance(obj, highs_var) == True:
            obj = highs_linear_expression(obj)

        if isinstance(obj, highs_linear_expression) == False or obj.sense != None:
            raise Exception('Objective cannot be an inequality') 

        self.update()
        super().changeObjectiveSense(ObjSense.kMaximize)
        
        # reset objective
        super().changeColsCost(self.numVars, range(self.numVars), [0]*self.numVars)

        # if we have duplicate variables, add the vals
        vars,vals = zip(*[(var, sum(v[1] for v in Vals)) for var, Vals in groupby(sorted(zip(obj.vars, obj.vals)), key=itemgetter(0))])
        super().changeColsCost(len(vars), vars, vals)
        super().changeObjectiveOffset(-obj.RHS)

        return super().run()

    
    # update variables
    def update(self):
        current_batch_size = len(self._batch.obj)

        if current_batch_size > 0:
            super().addVars(int(current_batch_size), self._batch.lb, self._batch.ub)
            super().changeColsCost(current_batch_size, self._batch.idx, self._batch.obj)
            super().changeColsIntegrality(current_batch_size, self._batch.idx, self._batch.type)

        self._batch = highs_batch(self)

    def val(self, var):
        return super().getSolution().col_value[var.index]

    def vals(self, vars):
        sol = super().getSolution()
        return [sol.col_value[v.index] for v in vars]


    #
    # add variable & useful constants
    #
    def addVar(self, lb = 0, ub = kHighsInf, obj = 0, type=HighsVarType.kContinuous, name = None):
        var = self._batch.add(obj, lb, ub, type, name, self)
        self._vars.append(var)
        return var

    def addIntegral(self, lb = 0, ub = kHighsInf, obj = 0, name = None):
        return self.addVar(lb, ub, obj, HighsVarType.kInteger, name)

    def addBinary(self, obj = 0, name = None):
        return self.addVar(0, 1, obj, HighsVarType.kInteger, name)

    def removeVar(self, var):
        for i in self._vars[var.index+1:]:
            i.index -= 1

        del self._vars[var.index]
        super().deleteVars(1, [var.index])

    def getVars(self):
        return self._vars

    @property
    def inf(self):
        return kHighsInf

    @property
    def numVars(self):
        return super().getNumCol()

    @property
    def numConstrs(self):
        return super().getNumRow()

    #
    # add constraints
    #
    def addConstr(self, cons, name=None):
        self.update()

        # if we have duplicate variables, add the vals
        vars,vals = zip(*[(var, sum(v[1] for v in Vals)) for var, Vals in groupby(sorted(zip(cons.vars, cons.vals)), key=itemgetter(0))])

        if cons.sense == '<':
            super().addRow(-self.inf, cons.RHS, len(vars), vars, vals)
        elif cons.sense == '>':
            super().addRow(cons.RHS, self.inf, len(vars), vars, vals)
        else:
            super().addRow(cons.RHS, cons.RHS, len(vars), vars, vals)

        cons = highs_cons(self.numConstrs - 1, self)
        self._cons.append(cons)
        return cons

    def chgCoeff(self, cons, var, val):
        super().changeCoeff(cons.index, var.index, val)

    def getConstrs(self):
        return self._cons

    def removeConstr(self, cons):
        for i in self._cons[cons.index+1:]:
            i.index -= 1

        del self._cons[cons.index]
        super().deleteRows(1, [cons.index])



## The following classes keep track of variables
## It is currently quite basic and may fail in complex scenarios

# highs variable
class highs_var(object):
    """Basic constraint builder for HiGHS"""
    __slots__ = ['index', '_varName', 'highs']

    def __init__(self, i, highs):
        self.index = i
        self.highs = highs
        self._varName = f"__v{i}"

    def __repr__(self):
        return f"{self._varName}"

    @property
    def name(self):
        return self._varName

    @name.setter
    def name(self, value):
        self._varName = value
        raise NotImplementedError()
        #self.highs.set_names(self.index, value)

    def __hash__(self):
        return self.index

    def __neg__(self):
        return -1.0 * highs_linear_expression(self)
    
    def __le__(self, other):
        return highs_linear_expression(self) <= other

    def __eq__(self, other):
        return highs_linear_expression(self) == other
    
    def __ge__(self, other):
        return highs_linear_expression(self) >= other

    def __add__(self, other):
        return highs_linear_expression(self) + other

    def __radd__(self, other):
        return highs_linear_expression(self) + other

    def __mul__(self, other):
        return highs_linear_expression(self) * other

    def __rmul__(self, other):
        return highs_linear_expression(self) * other

    def __rsub__(self, other):
        return -1.0 * highs_linear_expression(self) + other

    def __sub__(self, other):
        return highs_linear_expression(self) - other

# highs constraint
class highs_cons(object):
    """Basic constraint for HiGHS"""
    __slots__ = ['index', '_constrName', 'highs']
    
    def __init__(self, i, highs):
        self.index = i
        self.highs = highs
        self._constrName = f"__c{i}"

    def __repr__(self):
        return f"{self._constrName}"

    @property
    def name(self):
        return self._constrName

    @name.setter
    def name(self, value):
        self._constrName = value
        raise NotImplementedError()
        #self.highs.set_names(self.index, value)
    

# highs constraint builder
class highs_linear_expression(object):
    """Basic constraint builder for HiGHS"""
    __slots__ = ['vars', 'vals', 'RHS', 'sense']

    def __init__(self, var=None):
        if var:
            self.vars = [var.index]
            self.vals = [1.0]
        else:
            self.vars = []
            self.vals = []

        self.RHS = 0
        self.sense = None

    def __neg__(self):
        return -1.0 * self

    def __le__(self, other):
        self.sense = '<'

        if isinstance(other, highs_linear_expression):
            self.vars.extend(other.vars)
            self.vals.extend([-1.0 * v for v in other.vals])
            self.RHS -= other.RHS
            return self

        elif isinstance(other, highs_var):
            return NotImplemented

        elif isinstance(other, (int, float, Decimal)):
            self.RHS += other
            return self

        else:
            return NotImplemented

    def __eq__(self, other):
        self.sense = '='

        if isinstance(other, highs_linear_expression):
            self.vars.extend(other.vars)
            self.vals.extend([-1.0 * v for v in other.vals])
            self.RHS -= other.RHS
            return self

        elif isinstance(other, highs_var):
            return NotImplemented

        elif isinstance(other, (int, float, Decimal)):
            self.RHS += other
            return self

        else:
            return NotImplemented

    def __ge__(self, other):
        self.sense = '>'

        if isinstance(other, highs_linear_expression):
            self.vars.extend(other.vars)
            self.vals.extend([-1.0 * v for v in other.vals])
            self.RHS -= other.RHS
            return self

        elif isinstance(other, highs_var):
            return NotImplemented

        elif isinstance(other, (int, float, Decimal)):
            self.RHS += other
            return self

        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        if isinstance(other, highs_linear_expression):
            self.vars.extend(other.vars)
            self.vals.extend(other.vals)
            self.RHS += other.RHS
    
            if other.sense:
                self.sense = other.sense

            return self

        elif isinstance(other, highs_var):
            self.vars.append(other.index)
            self.vals.append(1.0)
            return self

        elif isinstance(other, (int, float, Decimal)):
            self.RHS -= other
            return self

        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if isinstance(other, (int, float, Decimal)):
            self.vals = [float(other) * v for v in self.vals]
            return self
        else:
            return NotImplemented

    def __rsub__(self, other):
        return other + -1.0 * self

    def __sub__(self, other):
        if isinstance(other, highs_linear_expression):
            return self + (-1.0 * other)
        elif isinstance(other, highs_var):
            return self + (-1.0 * highs_linear_expression(other))
        elif isinstance(other, (int, float, Decimal)):
            return self + (-1.0 * other)
        else:
            return NotImplemented

# used to batch add new variables
class highs_batch(object):
    """Batch constraint builder for HiGHS"""
    __slots__ = ['obj', 'lb', 'ub', 'type', 'name', 'highs', 'idx']

    def __init__(self, highs):
        self.highs = highs

        self.obj = []
        self.lb = []
        self.ub = []
        self.type = []
        self.name = []
        self.idx = []

    def add(self, obj, lb, ub, type,name,solver):
        self.obj.append(obj)
        self.lb.append(lb)
        self.ub.append(ub)
        self.type.append(type)
        self.name.append(name)

        newIndex = self.highs.numVars + len(self.obj)-1
        self.idx.append(newIndex)
        return highs_var(newIndex, solver)