import numpy as np
import math


# Target function
def target_func(var):
    return 2.0 * var[0]**2 + 1.0 * var[1]**2 + 3.0 * var[2]**2


# Area of projection -- 4x + y + z
def area():
    return np.array([4.0, 1.0, 1.0])


# Area of projection size -- 1 = 4x + y + z
def area_size():
    return 1.0


# Is point in area
def is_in_area(point):
    return 4.0 * point[0] + point[1] + point[2]


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


# Vector scalar
def scalar(var1, var2):
    return var1[0]*var2[0] + var1[1]*var2[1] + var1[2]*var2[2]


# Step grinding method
def step_grind(var, grad_var):
    alpha = 1.0
    while target_func(var) < target_func(var - alpha * grad_var):
        alpha /= 2.0
    return alpha


# Projection of point
def projection(var):
    return var + (area_size() - scalar(area(), var)) * area() / vector_norm(area())**2


# Print iteration
def print_iteration(i, alpha, var, var_pre):
    print ""
    print "ITERATION", i
    print "step_length =", alpha
    print "pr[x,y,z] =", var
    print "surface(pr[x,y,z]) =", is_in_area(var)
    print "f(pr[x,y,z]) = ", target_func(var)
    print "norm(pr[x,y,z] - pr[x,y,z]_prev) =", vector_norm(var - var_pre)


# Gradient method with projection
def gradient_method_with_projection(var0, eps, is_print=False):
    var = var0
    iteration = 0
    while True:
        var_previous = var
        alpha_current = step_grind(var, grad(var, eps))
        var_temp = var - alpha_current * grad(var, eps)
        var = projection(var_temp)
        iteration += 1
        if is_print:
            print_iteration(iteration, alpha_current, var, var_previous)
        if vector_norm(var - var_previous) < eps:
            break
    return var


def main():
    x0 = input("x0 = ")
    y0 = input("y0 = ")
    z0 = input("z0 = ")
    var0 = np.array([x0, y0, z0])
    res = gradient_method_with_projection(var0, 0.00001)
    print ""
    print "MIN =", res
    print "F min =", target_func(res)

main()