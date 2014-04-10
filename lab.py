import numpy as np
import math


# Target function
def target_func(var):
    return 2 * var[0]**2 + 1 * var[1]**2 + 3 * var[2]**2


# Derivative by parameter (0 for x, 1 for y, 2 for z)
def derivative_by(param, var, eps=0.001):
    var_plus_eps = np.array(var)
    var_plus_eps[param] = var[param] + eps
    var_minus_eps = np.array(var)
    var_minus_eps[param] = var[param] - eps
    result = target_func(var_plus_eps) - target_func(var_minus_eps)
    result /= 2 * eps
    return result


# Gradient of target function
def grad(var, eps=0.001):
    return np.array([derivative_by(0, var, eps), derivative_by(1, var, eps), derivative_by(2, var, eps)])


# Second derivative by same parameter (ex. d^2f/dx^2)
def sec_derivative_by_param(param, var, eps=0.001):
    var_plus_eps = np.array(var)
    var_plus_eps[param] = var[param] + eps
    var_minus_eps = np.array(var)
    var_minus_eps[param] = var[param] - eps
    result = target_func(var_plus_eps) - 2 * target_func(var) + target_func(var_minus_eps)
    result /= eps ** 2
    return result


# Second derivative by different parameters (ex. d^2f/dx*dz)
def sec_derivative_by_two_params(param1, param2, var, eps=0.001):
    var1_plus_eps = np.array(var)
    var1_plus_eps[param1] = var[param1] + eps
    var2_plus_eps = np.array(var)
    var2_plus_eps[param2] = var[param2] + eps
    var_both_plus_eps = np.array(var)
    var_both_plus_eps[param1] = var1_plus_eps[param1]
    var_both_plus_eps[param2] = var2_plus_eps[param2]
    result = target_func(var_both_plus_eps) - target_func(var1_plus_eps) - target_func(var2_plus_eps) + target_func(var)
    result /= eps ** 2
    return result


# Vector norm
def vector_norm(var):
    return math.sqrt(var[0]**2 + var[1]**2 + var[2]**2)


# Step grinding method
def step_grind(var, grad_var):
    alpha = 100.0
    while target_func(var) <= target_func(var - alpha * grad_var):
        alpha /= 2.0
    return alpha


# Gradient method
def gradient_method(var0, eps, print_iterations=False):
    var = var0
    iteration = 0
    while vector_norm(grad(var, eps)) >= eps:
        alpha_current = step_grind(var, grad(var, eps))
        var -= alpha_current * grad(var, eps)
        iteration += 1
        if print_iterations:
            print ""
            print "Iteration", iteration
            print "step_length =", alpha_current
            print "[x,y,z] =", var
            print "norm(grad) =", vector_norm(grad(var, eps))
    print ""
    print ""
    print "MIN =", var
    return var


def main():
    x0 = input("x0 = ")
    y0 = input("y0 = ")
    z0 = input("z0 = ")
    var0 = np.array([x0, y0, z0])
    gradient_method(var0, 0.0001)

main()