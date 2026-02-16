import numpy as np
import spiceypy as spy


def computeNumericalJacobian(jfun, x, dx, **args):
    """
    Computes numerically the Jacobian matrix of a multivariate function.

    Parameters:
        jfun: multivariate function with the prototype "def jfun(x,**args)", function
        x: indepedent variables, numpy array (N).
        dx: step size of independent variables, numpy array (N).
        **args: argument of the function

    Return:
        y: dependent variables, y=jfun(x,**args)
        Jyx: Jacobian matrix:

            Jif= [dy_1/dx_1,dy_1/dx_2,...,dy_1/dx_N,
                dy_2/dx_1,dy_2/dx_2,...,dy_2/dx_N,
                                . . .
                dy_N/dx_1,dy_N/dx_2,...,dy_N/dx_N,]
    """
    N = len(x)
    J = np.zeros((N, N))
    y = jfun(x, **args)
    for i in range(N):
        for j in range(N):
            pre = [x[k] for k in range(j)]
            pos = [x[k] for k in range(j + 1, N)]
            yi = lambda t: jfun(pre + [t] + pos, **args)[i]
            dyidxj = (yi(x[j] + dx[j]) - yi(x[j] - dx[j])) / (2 * dx[j])
            J[i, j] = dyidxj
    return y, J
