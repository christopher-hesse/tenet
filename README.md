# tenet

A [torch](https://github.com/pytorch/pytorch)-inspired automatic differentiation prototype for [Zig](https://ziglang.org/).

Imagine the [numpy](https://numpy.org/) NDArray, only you can also compute backward in time using inverted functions.  Well, not quite, but you *can* calculate derivatives with respect to the inputs of your computation.

## Usage

The main struct is `Tensor`, an N-dimensional array of numbers, usually floating point numbers.  Here's a short example showing how to do a `+` operation along with a backward pass:

```zig
const tenet = @import("tenet.zig");
const alc = std.testing.allocator;
var a = try tenet.Tensor.allocWithValue(f32, alc, &[_]u64{2, 3, 4}, 1.0, tenet.tensor.REQUIRES_GRAD);
defer a.release();
var b = try tenet.Tensor.allocWithValue(f32, alc, &[_]u64{2, 3, 4}, 2.0, tenet.tensor.REQUIRES_GRAD);
defer b.release();
var out = try tenet.tensor.plusAlloc(alc, a, b);
defer out.release();
var grad_out = try tenet.Tensor.allocWithValue(f32, alc, &[_]u64{2, 3, 4}, 4.0, 0);
defer grad_out.release();
try tenet.tensor.backwardAlloc(alc, out, grad_out);
std.testing.expect(tenet.array.equal(a.grad.?, grad_out.data));
std.testing.expect(tenet.array.equal(b.grad.?, grad_out.data));
```

For a full example, look at the [MNIST example](src/main.zig).

## Automatic Differentiation

If you have a function `z = f(x, y)` and you want to know how to change `x` and `y` to minimize `z`, how do you do find that out?  One way would be to increase and decrease `x` and `y` individually to see how much `z` changes, then move them in whichever direction is better.  That method is called ["finite differences"](https://en.wikipedia.org/wiki/Finite_difference#Relation_with_derivatives).

For a couple of input variables, this is fine, but it's not very efficient with a large number of input variables.  Instead of doing that, you can find the derivatives by constructing a sort of backward version of the computation graph of your function.  If the function `f` looked like this:

```py
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
```

You might have a backward function like this:

```py
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
```

Where the `backward_` functions are the derivatives of the original functions, using the chain rule to combine them together.  Each `backward_` function takes the original inputs to the normal function, plus an extra `grad_out` parameter, then returns `grad_in_<name>` for each of the original inputs.  You end up with the same information about how the output changes as you would get from changing each input variable individually, only with fewer calculations:

```py
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
```

See [scripts/grad_example.py](scripts/grad_example.py) for the full script.  In the case where the inputs and outputs are matrices instead of scalars, `grad_out` will have the shape of the output, and each `grad_in_<name>` will have the shape of the corresponding input.

In automatic differentiation, you create `backward_f` automatically based on the operations done by `f`.  Like in torch, no explicit graph is defined when using this prototype.  Arrays in `tenet` track the series of operations used to create them, so when you do the backward pass, each `backward_` function is run for you, automatically.

## Interesting Features

There's only one sort of interesting feature about this prototype.  Zig does not support operator overloading, but it would still be nice to write out equations.  Writing out the operations by hand is a bit of a pain:

```zig
// (x * y + z) ^ 2.0
var a = try multiplyAlloc(alc, x, y);
defer a.release();
var b = try addAlloc(alc, a, z);
defer b.release()
var two = try Tensor.allocWithValue(f32, alc, &[_]u64{}, 2, tensor.NO_FLAGS);
defer two.release();
var c = try powerAlloc(alc, b, two);
defer c.release();
```

The `expr` function does all the same stuff, but uses a string at compile time:

```zig
var c = try expr(alc, "(x .* y + z) .^ 2.0", .{.x=x, .y=y, .z=z});
defer c.release();
```

Actually it only parses the expression at compile time, it doesn't fully unroll all the operations. I suspect the only thing keeping it from fully unrolling is some Zig compiler bug.

Because operator overloading is not used, the `expr` syntax has much fewer limitations.  For this prototype, it uses [MATLAB style operators](https://www.mathworks.com/help/matlab/matlab_prog/matlab-operators-and-special-characters.html).

## Downsides

* Defining an explicit graph may be a better approach than this and is used in the [kann](https://github.com/attractivechaos/kann) library
* Deallocating memory immediately is kind of annoying when you don't use `expr`.  If you use `defer`, it won't be deallocated until the end of the block
* Performance is mediocre, there has been no tuning for performance beyond an option to use [Intel's MKL library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html#gs.zou9ms).  The option just causes the library to be linked, you still have to enable it manually in the code.
* CPU only for now
* Only tested on windows
* Probably contains serious bugs
* This is mostly a proof-of-concept, and will likely not be maintained as a generally useful library.
