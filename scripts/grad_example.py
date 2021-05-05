def square(x):
    return x ** 2


def cube(x):
    return x ** 3


def multiply(x, y):
    return x * y


def f(x, y):
    a = square(x)
    b = cube(y)
    c = multiply(a, b)
    return c


def backward_multiply(x, y, grad_out):
    grad_in_x = y * grad_out
    grad_in_y = x * grad_out
    return grad_in_x, grad_in_y


def backward_square(x, grad_out):
    grad_in = 2 * x * grad_out
    return grad_in


def backward_cube(x, grad_out):
    grad_in = 3 * x ** 2 * grad_out
    return grad_in


def backward_f(x, y, grad_z):
    # we actually need the intermediate values to call the backward functions
    # so re-calculate them here (normally we would just store them when running f() the first time)
    a = square(x)
    b = cube(y)
    _c = multiply(a, b)

    grad_a, grad_b = backward_multiply(a, b, grad_z)
    grad_y = backward_cube(y, grad_b)
    grad_x = backward_square(x, grad_a)
    return grad_x, grad_y


# run the function normally
x = 1.0
y = 2.0
z = f(x, y)
print(f"f(x,y): {z}")

# run the backward function
grad_z = 1.0  # the initial grad value is set to 1
grad_x, grad_y = backward_f(x, y, grad_z)
print(f"backward_f(x, y, grad_z): grad_x = {grad_x}, grad_y = {grad_y}")

# check the backward function using finite differences
# by making small changes to each input to find how the output changes
def finite_differences(x, y, f, epsilon=1e-6):
    grad_x = (f(x + epsilon, y) - f(x - epsilon, y)) / (2 * epsilon)
    grad_y = (f(x, y + epsilon) - f(x, y - epsilon)) / (2 * epsilon)
    return grad_x, grad_y


grad_x_fd, grad_y_fd = finite_differences(x, y, f)
print(f"finite differences approximation: grad_x = {grad_x_fd}, grad_y = {grad_y_fd}")
