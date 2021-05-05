// A wrapper for an N-dimensional array that keeps track of the computation graph
// so that gradients can be calculated with respect to the inputs of the graph.
//
// The underlying array is available as the .data property and, if REQUIRES_GRAD
// is set for the Tensor, the .grad property will contain a second array the same
// shape and dtype as the array. The gradient will be placed in that array
// when it is calculated using backwardAlloc().
//
// These are reference counted, so you can call .retain() and .release() on them.
//
// Tensors can be provided with the data and grad arrays, but the following
// things will still be allocated:
//
//   grad_record: these are allocated whenever the Tensor is created as the result of an operation
//                      and deallocated when the Tensor is deallocated
//   ref_counter: keeps track of the reference count of the Tensor, which is separate from the
//                      reference counts of the data and grad arrays.  When the reference count
//                      hits zero, the data and grad will have their reference counts decremented by 1.

const std = @import("std");
const array = @import("array.zig");
const Array = array.Array;
const DType = array.DType;
const reference_counter = @import("reference_counter.zig");
const ReferenceCounter = reference_counter.ReferenceCounter;

pub const NO_FLAGS = 0;
pub const REQUIRES_GRAD = 1;
pub const IS_BRANCH = 2;

const ForwardError = error{OutOfMemory};

const BackwardError = error{OutOfMemory};

const ForwardFn = fn (alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array;

const BackwardFn = fn (alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void;

const DeallocFn = fn (alc: *std.mem.Allocator, extra_args_ptr: u64) void;

const GradientRecord = struct {
    inputs: []Tensor,
    extra_args_ptr: u64,
    output: Tensor,
    grad_output: ?Array,
    backward_fn: BackwardFn,
    dealloc_fn: ?DeallocFn,
    alc: *std.mem.Allocator,

    const Self = @This();

    pub fn alloc(alc: *std.mem.Allocator, inputs: []Tensor, extra_args_ptr: u64, output: Tensor, backward_fn: BackwardFn, dealloc_fn: ?DeallocFn) !Self {
        var i = try alc.alloc(Tensor, inputs.len);
        errdefer alc.free(i);
        std.mem.copy(Tensor, i, inputs);
        for (i) |*t| {
            t.retain();
        }
        // we don't keep a reference to the output tensor, since the output tensor will
        // own this GradientRecord
        return Self{ .inputs = i, .extra_args_ptr = extra_args_ptr, .output = output, .grad_output = null, .alc = alc, .backward_fn = backward_fn, .dealloc_fn = dealloc_fn };
    }

    pub fn dealloc(self: *Self) void {
        for (self.inputs) |*t| {
            t.release();
        }
        if (self.grad_output != null) {
            @panic("grad_output present on GradientRecord");
        }
        self.alc.free(self.inputs);
        if (self.dealloc_fn) |dealloc_fn| {
            dealloc_fn(self.alc, self.extra_args_ptr);
        }
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        if (fmt.len != 0) {
            @compileError("Unknown format character: '" ++ f ++ "'");
        }
        try std.fmt.format(writer, "GradientRecord(num_inputs={}, has_grad_output={})", .{ self.inputs.len, self.grad_output != null });
    }
};

fn has_flag(flags: u64, flag: u64) bool {
    return flags & flag == flag;
}

pub const Tensor = struct {
    data: Array,
    grad: ?Array,
    requires_grad: bool,
    is_leaf: bool,
    grad_record: ?*GradientRecord,
    ref_counter: ?*ReferenceCounter,
    alc: ?*std.mem.Allocator,

    const Self = @This();

    fn alloc(alc: *std.mem.Allocator, data: Array, grad: ?Array, flags: u64) !Self {
        var requires_grad = has_flag(flags, REQUIRES_GRAD);
        var is_leaf = !has_flag(flags, IS_BRANCH);
        if (requires_grad and !(data.dtype == array.DType.f32 or data.dtype == array.DType.f64)) {
            @panic("grad requires floating point dtype");
        }
        var grad_array: ?Array = grad;
        if (is_leaf and requires_grad and grad == null) {
            grad_array = try array.zerosLikeAlloc(alc, data);
        }
        var ref_counter = try alc.create(ReferenceCounter);
        ref_counter.* = ReferenceCounter.init();
        return Self{ .data = data, .grad = grad_array, .requires_grad = requires_grad, .is_leaf = is_leaf, .grad_record = null, .ref_counter = ref_counter, .alc = alc };
    }

    pub fn allocWithValue(comptime T: type, alc: *std.mem.Allocator, shape: []const u64, value: T, flags: u64) !Self {
        var data = try Array.allocWithValue(T, alc, shape, value);
        return Self.alloc(alc, data, null, flags);
    }

    pub fn allocWithString(comptime T: type, alc: *std.mem.Allocator, str: []const u8, flags: u64) !Self {
        var data = try Array.allocWithString(T, alc, str);
        return Self.alloc(alc, data, null, flags);
    }

    pub fn allocWithRange(comptime T: type, alc: *std.mem.Allocator, shape: []const u64, start: T, step: T, flags: u64) !Self {
        var data = try Array.allocWithRange(T, alc, shape, start, step);
        return Self.alloc(alc, data, null, flags);
    }

    pub fn allocWithData(alc: *std.mem.Allocator, data: Array, flags: u64) !Self {
        data.retain();
        return Self.alloc(alc, data, null, flags);
    }

    pub fn allocWithDataAndGrad(alc: *std.mem.Allocator, data: Array, grad: Array, flags: u64) !Self {
        data.retain();
        grad.retain();
        if (!has_flag(flags, REQUIRES_GRAD)) {
            @panic("must require grad if grad is specified");
        }
        array.assertShapesAreTheSame(data, grad);
        array.assertTypesAreTheSame(data, grad);
        return Self.alloc(alc, data, grad, flags);
    }

    pub fn allocWithBuffers(comptime T: type, alc: *std.mem.Allocator, shape: []const u64, data_buf: []T, grad_buf: []T) !Self {
        var data = Array.fromBuffer(T, shape, data_buf);
        var grad = Array.fromBuffer(T, shape, grad_buf);
        return Self.alloc(alc, data, grad, REQUIRES_GRAD);
    }

    pub fn fromBuffer(comptime T: type, shape: []const u64, data_buf: []T) Self {
        var data = Array.fromBuffer(T, shape, data_buf);
        return Self{ .data = data, .grad = null, .requires_grad = false, .is_leaf = true, .grad_record = null, .ref_counter = null, .alc = null };
    }

    pub fn flatFromBuffer(comptime T: type, data_buf: []T) Self {
        return Self.fromBuffer(T, &[_]u64{data_buf.len}, data_buf);
    }

    pub fn scalarFromBuffer(comptime T: type, data_buf: []T) Self {
        return Self.fromBuffer(T, &[_]u64{}, data_buf);
    }

    pub fn getDType(self: Self) DType {
        return self.data.dtype;
    }

    pub fn narrowView(self: Self, pos: []const u64, shape: []const u64) Self {
        var grad = self.grad;
        if (grad != null) {
            grad = grad.?.narrowView(pos, shape);
        }
        return Self{ .data = self.data.narrowView(pos, shape), .grad = grad, .requires_grad = self.requires_grad, .is_leaf = self.is_leaf, .grad_record = self.grad_record, .ref_counter = self.ref_counter, .alc = self.alc };
    }

    pub fn reshapeView(self: Self, shape: []const u64) Self {
        var grad = self.grad;
        if (grad != null) {
            grad = grad.?.reshapeView(shape);
        }
        return Self{ .data = self.data.reshapeView(shape), .grad = grad, .requires_grad = self.requires_grad, .is_leaf = self.is_leaf, .grad_record = self.grad_record, .ref_counter = self.ref_counter, .alc = self.alc };
    }

    pub fn retain(self: Self) void {
        if (self.ref_counter) |ref_counter| {
            ref_counter.increment();
        }
    }

    pub fn release(self: Self) void {
        if (self.ref_counter) |ref_counter| {
            if (ref_counter.decrement()) {
                self.data.release();
                if (self.grad) |g| {
                    g.release();
                }
                var alc = self.alc.?;
                alc.destroy(ref_counter);
                // if the tensor has a grad record, it's owned by the tensor
                if (self.grad_record) |gr| {
                    gr.dealloc();
                    alc.destroy(gr);
                }
            }
        }
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        if (fmt.len != 0) {
            @compileError("Unknown format character: '" ++ f ++ "'");
        }
        try std.fmt.format(writer, "Tensor(is_leaf={}, requires_grad={}, data={}, grad={}, grad_record={})", .{ self.is_leaf, self.requires_grad, self.data, self.grad, self.grad_record });
    }
};

test "format_tensor" {
    var t = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 1.0, REQUIRES_GRAD);
    defer t.release();
    var t2 = try timesAlloc(std.testing.allocator, t, t);
    defer t2.release();
    std.debug.print("{}\n", .{t2});
}

pub fn zerosAlloc(alc: *std.mem.Allocator, dtype: DType, shape: []const u64, flags: u64) !Tensor {
    var data = try array.zerosAlloc(alc, dtype, shape);
    var t = try Tensor.allocWithData(alc, data, flags);
    data.release();
    return t;
}

pub fn zerosLikeAlloc(alc: *std.mem.Allocator, t: Tensor, flags: u64) !Tensor {
    return zerosAlloc(alc, t.getDType(), t.data.getShape(), flags);
}

pub fn onesAlloc(alc: *std.mem.Allocator, dtype: DType, shape: []const u64, flags: u64) !Tensor {
    var data = try array.onesAlloc(alc, dtype, shape);
    var t = try Tensor.allocWithData(alc, data, flags);
    data.release();
    return t;
}

pub fn onesLikeAlloc(alc: *std.mem.Allocator, t: Tensor, flags: u64) !Tensor {
    return onesAlloc(alc, t.getDType(), t.data.getShape(), flags);
}

pub fn detachAlloc(alc: *std.mem.Allocator, x: Tensor) !Tensor {
    return try Tensor.allocWithData(alc, x.data, NO_FLAGS);
}

pub fn scalarAlloc(alc: *std.mem.Allocator, dtype: DType, value: f64) !Tensor {
    var data = try array.scalarAlloc(alc, dtype, value);
    var output = try Tensor.allocWithData(alc, data, 0);
    data.release();
    return output;
}

pub fn plusAlloc(alc: *std.mem.Allocator, x: Tensor, y: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{ x, y }, &[_]Array{ x.data, y.data }, 0, plusForwardAlloc, plusBackwardAlloc, null);
}

pub fn plusForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.plusAlloc(alc, inputs[0], inputs[1]);
}

pub fn plusBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    for (grad_inputs_out) |maybe_grad_input| {
        if (maybe_grad_input) |grad_input| {
            array.bcastsum(grad_output, grad_input);
        }
    }
}

pub fn uplusAlloc(alc: *std.mem.Allocator, x: Tensor) !Tensor {
    x.retain();
    return x;
}

pub fn minusAlloc(alc: *std.mem.Allocator, x: Tensor, y: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{ x, y }, &[_]Array{ x.data, y.data }, 0, minusForwardAlloc, minusBackwardAlloc, null);
}

pub fn minusForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.minusAlloc(alc, inputs[0], inputs[1]);
}

pub fn minusBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    for (grad_inputs_out) |maybe_grad_input, i| {
        if (maybe_grad_input) |grad_input| {
            array.bcastsum(grad_output, grad_input);
            if (i == 1) {
                var negative_one = try array.scalarAlloc(alc, grad_input.dtype, -1);
                defer negative_one.release();
                array.times(grad_input, negative_one, grad_input);
            }
        }
    }
}

pub fn logAlloc(alc: *std.mem.Allocator, x: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{x}, &[_]Array{x.data}, 0, logForwardAlloc, logBackwardAlloc, null);
}

pub fn logForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.logAlloc(alc, inputs[0]);
}

pub fn logBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (grad_inputs_out[0]) |grad| {
        var x = inputs[0];
        var grad_input = try array.expr(alc, "g ./ x", .{ .x = x, .g = grad_output });
        defer grad_input.release();
        array.copy(grad_input, grad);
    }
}

test "log_gradcheck" {
    var a = try Tensor.allocWithRange(f64, std.testing.allocator, &[_]u64{ 3, 4 }, 1.0, 1.0, REQUIRES_GRAD);
    defer a.release();
    var inputs = [_]Tensor{a};
    std.testing.expect(try gradCheck(f64, std.testing.allocator, logForwardAlloc, logBackwardAlloc, &inputs, 0));
}

pub fn log2Alloc(alc: *std.mem.Allocator, x: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{x}, &[_]Array{x.data}, 0, log2ForwardAlloc, log2BackwardAlloc, null);
}

pub fn log2ForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.log2Alloc(alc, inputs[0]);
}

pub fn log2BackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (grad_inputs_out[0]) |grad| {
        var x = inputs[0];
        var grad_input = try array.expr(alc, "1.0 ./ (x .* log(2.0)) .* g", .{ .x = x, .g = grad_output });
        defer grad_input.release();
        array.copy(grad_input, grad);
    }
}

test "log2_gradcheck" {
    var a = try Tensor.allocWithRange(f64, std.testing.allocator, &[_]u64{ 3, 4 }, 1.0, 1.0, REQUIRES_GRAD);
    defer a.release();
    var inputs = [_]Tensor{a};
    std.testing.expect(try gradCheck(f64, std.testing.allocator, log2ForwardAlloc, log2BackwardAlloc, &inputs, 0));
}

pub fn expAlloc(alc: *std.mem.Allocator, x: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{x}, &[_]Array{x.data}, 0, expForwardAlloc, expBackwardAlloc, null);
}

pub fn expForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.expAlloc(alc, inputs[0]);
}

pub fn expBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (grad_inputs_out[0]) |grad| {
        var x = inputs[0];
        var grad_input = try array.expr(alc, "exp(x) .* g", .{ .x = x, .g = grad_output });
        defer grad_input.release();
        array.copy(grad_input, grad);
    }
}

test "exp_gradcheck" {
    var a = try Tensor.allocWithRange(f64, std.testing.allocator, &[_]u64{ 3, 4 }, 1.0, 1.0, REQUIRES_GRAD);
    defer a.release();
    var inputs = [_]Tensor{a};
    std.testing.expect(try gradCheck(f64, std.testing.allocator, expForwardAlloc, expBackwardAlloc, &inputs, 0));
}

pub fn uminusAlloc(alc: *std.mem.Allocator, x: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{x}, &[_]Array{x.data}, 0, uminusForwardAlloc, uminusBackwardAlloc, null);
}

pub fn uminusForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.uminusAlloc(alc, inputs[0]);
}

pub fn uminusBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (grad_inputs_out[0]) |grad_input| {
        array.copy(grad_output, grad_input);
        var negative_one = try array.scalarAlloc(alc, grad_input.dtype, -1);
        defer negative_one.release();
        array.times(grad_input, negative_one, grad_input);
    }
}

pub fn timesAlloc(alc: *std.mem.Allocator, x: Tensor, y: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{ x, y }, &[_]Array{ x.data, y.data }, 0, timesForwardAlloc, timesBackwardAlloc, null);
}

pub fn timesForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.timesAlloc(alc, inputs[0], inputs[1]);
}

pub fn timesBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (inputs.len != 2) {
        @panic("invalid number of inputs");
    }

    var x = inputs[0];
    var y = inputs[1];

    var x_grad_to_sum = try array.timesAlloc(alc, grad_output, y);
    defer x_grad_to_sum.release();
    if (grad_inputs_out[0]) |x_grad| {
        array.bcastsum(x_grad_to_sum, x_grad);
    }

    var y_grad_to_sum = try array.timesAlloc(alc, grad_output, x);
    defer y_grad_to_sum.release();
    if (grad_inputs_out[1]) |y_grad| {
        array.bcastsum(y_grad_to_sum, y_grad);
    }
}

pub fn mtimesAlloc(alc: *std.mem.Allocator, x: Tensor, y: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{ x, y }, &[_]Array{ x.data, y.data }, 0, mtimesForwardAlloc, mtimesBackwardAlloc, null);
}

pub fn mtimesForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.mtimesAlloc(alc, inputs[0], inputs[1]);
}

pub fn mtimesBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (inputs.len != 2) {
        @panic("invalid number of inputs");
    }

    var x = inputs[0];
    var y = inputs[1];

    if (grad_inputs_out[0]) |x_grad| {
        var x_grad_to_copy = try array.expr(alc, "g * y'", .{ .y = y, .g = grad_output });
        defer x_grad_to_copy.release();
        array.copy(x_grad_to_copy, x_grad);
    }

    if (grad_inputs_out[1]) |y_grad| {
        var y_grad_to_copy = try array.expr(alc, "x' * g", .{ .x = x, .g = grad_output });
        defer y_grad_to_copy.release();
        array.copy(y_grad_to_copy, y_grad);
    }
}

test "mtimes_gradcheck" {
    var a = try Tensor.allocWithRange(f64, std.testing.allocator, &[_]u64{ 3, 4 }, 0.0, 1.0, REQUIRES_GRAD);
    defer a.release();
    var b = try Tensor.allocWithRange(f64, std.testing.allocator, &[_]u64{ 4, 3 }, 1.0, 2.0, REQUIRES_GRAD);
    defer b.release();
    var inputs = [_]Tensor{ a, b };
    std.testing.expect(try gradCheck(f64, std.testing.allocator, mtimesForwardAlloc, mtimesBackwardAlloc, &inputs, 0));
}

pub fn divideAlloc(alc: *std.mem.Allocator, x: Tensor, y: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{ x, y }, &[_]Array{ x.data, y.data }, 0, divideForwardAlloc, divideBackwardAlloc, null);
}

pub fn divideForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.divideAlloc(alc, inputs[0], inputs[1]);
}

pub fn divideBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (inputs.len != 2) {
        @panic("invalid number of inputs");
    }

    var x = inputs[0];
    var y = inputs[1];

    if (grad_inputs_out[0]) |x_grad| {
        var x_grad_to_sum = try array.divideAlloc(alc, grad_output, y);
        defer x_grad_to_sum.release();
        array.bcastsum(x_grad_to_sum, x_grad);
    }

    if (grad_inputs_out[1]) |y_grad| {
        var y_grad_to_sum = try array.expr(alc, "-x ./ (y .* y) .* g", .{ .x = x, .y = y, .g = grad_output });
        defer y_grad_to_sum.release();
        array.bcastsum(y_grad_to_sum, y_grad);
    }
}

pub fn powerAlloc(alc: *std.mem.Allocator, x: Tensor, y: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{ x, y }, &[_]Array{ x.data, y.data }, 0, powerForwardAlloc, powerBackwardAlloc, null);
}

pub fn powerForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.powerAlloc(alc, inputs[0], inputs[1]);
}

pub fn powerBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (inputs.len != 2) {
        @panic("invalid number of inputs");
    }

    var x = inputs[0];
    var y = inputs[1];

    if (grad_inputs_out[0]) |x_grad| {
        var x_grad_to_sum = try array.expr(alc, "y .* (x .^ (y-1)) .* g", .{ .x = x, .y = y, .g = grad_output });
        defer x_grad_to_sum.release();
        array.bcastsum(x_grad_to_sum, x_grad);
    }

    if (grad_inputs_out[1]) |y_grad| {
        var y_grad_to_sum = try array.expr(alc, "log(x) .* (x .^ y) .* g", .{ .x = x, .y = y, .g = grad_output });
        defer y_grad_to_sum.release();
        array.bcastsum(y_grad_to_sum, y_grad);
    }
}

pub fn maxAlloc(alc: *std.mem.Allocator, x: Tensor, y: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{ x, y }, &[_]Array{ x.data, y.data }, 0, maxForwardAlloc, maxBackwardAlloc, null);
}

pub fn maxForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return try array.maxAlloc(alc, inputs[0], inputs[1]);
}

pub fn maxBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (inputs.len != 2) {
        @panic("invalid number of inputs");
    }

    var x = inputs[0];
    var y = inputs[1];

    if (grad_inputs_out[0]) |x_grad| {
        var x_grad_to_sum = try array.expr(alc, "(x > y) .* g", .{ .x = x, .y = y, .g = grad_output });
        defer x_grad_to_sum.release();
        array.bcastsum(x_grad_to_sum, x_grad);
    }

    if (grad_inputs_out[1]) |y_grad| {
        var y_grad_to_sum = try array.expr(alc, "(y >= x) .* g", .{ .x = x, .y = y, .g = grad_output });
        defer y_grad_to_sum.release();
        array.bcastsum(y_grad_to_sum, y_grad);
    }
}

test "max_gradcheck" {
    // gradcheck doesn't work for max because it is discontinuous, so just check that the behavior is similar to pytorch
    // torch.max(a, b)
    // a.grad tensor([1., 0., 0.], dtype=torch.float64)
    // b.grad tensor([0., 1., 1.], dtype=torch.float64)
    // torch.max(b, a)
    // a.grad tensor([1., 0., 1.], dtype=torch.float64)
    // b.grad tensor([0., 1., 0.], dtype=torch.float64)
    {
        var a_buf = [_]f32{ 0.0, 0.0, 2.0 };
        var a_grad_buf = [_]f32{ 0.0, 0.0, 0.0 };
        const a = try Tensor.allocWithBuffers(f32, std.testing.allocator, &[_]u64{3}, &a_buf, &a_grad_buf);
        defer a.release();
        var b_buf = [_]f32{ -1.0, 1.0, 2.0 };
        var b_grad_buf = [_]f32{ 0.0, 0.0, 0.0 };
        const b = try Tensor.allocWithBuffers(f32, std.testing.allocator, &[_]u64{3}, &b_buf, &b_grad_buf);
        defer b.release();
        const c = try maxAlloc(std.testing.allocator, a, b);
        defer c.release();
        var d = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{3}, 3.0, 0);
        defer d.release();
        try backwardAlloc(std.testing.allocator, c, d);
        var a_grad_expected_buf = [_]f32{ 3.0, 0.0, 0.0 };
        const a_grad_expected = Array.fromBuffer(f32, &[_]u64{3}, &a_grad_expected_buf);
        var b_grad_expected_buf = [_]f32{ 0.0, 3.0, 3.0 };
        const b_grad_expected = Array.fromBuffer(f32, &[_]u64{3}, &b_grad_expected_buf);
        std.testing.expect(array.equal(a.grad.?, a_grad_expected));
        std.testing.expect(array.equal(b.grad.?, b_grad_expected));
    }

    {
        var a_buf = [_]f32{ 0.0, 0.0, 2.0 };
        var a_grad_buf = [_]f32{ 0.0, 0.0, 0.0 };
        const a = try Tensor.allocWithBuffers(f32, std.testing.allocator, &[_]u64{3}, &a_buf, &a_grad_buf);
        defer a.release();
        var b_buf = [_]f32{ -1.0, 1.0, 2.0 };
        var b_grad_buf = [_]f32{ 0.0, 0.0, 0.0 };
        const b = try Tensor.allocWithBuffers(f32, std.testing.allocator, &[_]u64{3}, &b_buf, &b_grad_buf);
        defer b.release();
        const c = try maxAlloc(std.testing.allocator, b, a);
        defer c.release();
        var d = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{3}, 3.0, 0);
        defer d.release();
        try backwardAlloc(std.testing.allocator, c, d);
        var a_grad_expected_buf = [_]f32{ 3.0, 0.0, 3.0 };
        const a_grad_expected = Array.fromBuffer(f32, &[_]u64{3}, &a_grad_expected_buf);
        var b_grad_expected_buf = [_]f32{ 0.0, 3.0, 0.0 };
        const b_grad_expected = Array.fromBuffer(f32, &[_]u64{3}, &b_grad_expected_buf);
        std.testing.expect(array.equal(a.grad.?, a_grad_expected));
        std.testing.expect(array.equal(b.grad.?, b_grad_expected));
    }
}

const CastArgs = struct {
    dtype: DType,
};

pub fn castAlloc(alc: *std.mem.Allocator, x: Tensor, dtype: DType) !Tensor {
    if (x.data.dtype == dtype) {
        x.retain();
        return x;
    }
    // we only need this temporarily for the autograd interface, the backward pass can figure the dtype out
    var cast_args = CastArgs{
        .dtype = dtype,
    };
    var result = autogradAlloc(alc, &[_]Tensor{x}, &[_]Array{x.data}, @ptrToInt(&cast_args), castForwardAlloc, castBackwardAlloc, null);
    return result;
}

pub fn castForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    var cast_args_ptr = @intToPtr(*CastArgs, extra_args_ptr);
    return try array.castAlloc(alc, inputs[0], cast_args_ptr.dtype);
}

pub fn castBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (inputs.len != 1) {
        @panic("invalid number of inputs");
    }

    if (grad_inputs_out[0]) |grad| {
        array.cast(grad_output, grad);
    }
}

test "cast_grad" {
    var a = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 3 }, 1.0, REQUIRES_GRAD);
    defer a.release();
    var b = try castAlloc(std.testing.allocator, a, DType.f32);
    defer b.release();
    var c = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 3.0, 0);
    defer c.release();
    var d = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 3 }, 3.0, 0);
    defer d.release();
    try backwardAlloc(std.testing.allocator, b, c);
    std.testing.expect(array.equal(a.grad.?, d.data));
}

const GatherArgs = struct {
    dim: u64,
};

pub fn gatherAlloc(alc: *std.mem.Allocator, x: Tensor, dim: u64, index: Tensor) !Tensor {
    var args = try alc.create(GatherArgs);
    args.* = GatherArgs{
        .dim = dim,
    };
    if (index.requires_grad) {
        @panic("Index cannot have requires_grad set");
    }
    return autogradAlloc(alc, &[_]Tensor{ x, index }, &[_]Array{ x.data, index.data }, @ptrToInt(args), gatherForwardAlloc, gatherBackwardAlloc, gatherDealloc);
}

pub fn gatherForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    var args_ptr = @intToPtr(*GatherArgs, extra_args_ptr);
    return try array.gatherAlloc(alc, inputs[0], args_ptr.dim, inputs[1]);
}

pub fn gatherBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (inputs.len != 2) {
        @panic("invalid number of inputs");
    }

    var index = inputs[1];
    var args_ptr = @intToPtr(*GatherArgs, extra_args_ptr);
    var dim = args_ptr.dim;
    if (grad_inputs_out[0]) |grad| {
        array.scatter(grad_output, grad, dim, index);
    }
}

fn gatherDealloc(alc: *std.mem.Allocator, extra_args_ptr: u64) void {
    var args = @intToPtr(*GatherArgs, extra_args_ptr);
    alc.destroy(args);
}

test "gather_grad" {
    const input = try Tensor.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0, REQUIRES_GRAD);
    defer input.release();
    const index = try Tensor.allocWithString(u64, std.testing.allocator, "[[[0], [1], [2]], [[3], [2], [1]]]", NO_FLAGS);
    defer index.release();
    var output = try gatherAlloc(std.testing.allocator, input, 2, index);
    defer output.release();
    const expected_output = try Tensor.allocWithString(f32, std.testing.allocator, "[[[0], [5], [10]], [[15], [18], [21]]]", NO_FLAGS);
    defer expected_output.release();
    std.testing.expect(array.equal(output.data, expected_output.data));
    const grad_output = try Tensor.allocWithValue(f32, std.testing.allocator, output.data.getShape(), 3.0, NO_FLAGS);
    defer grad_output.release();
    try backwardAlloc(std.testing.allocator, output, grad_output);
    const expected_grad = try Tensor.allocWithString(f32, std.testing.allocator, "[[[3,0,0,0], [0,3,0,0], [0,0,3,0]], [[0,0,0,3], [0,0,3,0], [0,3,0,0]]]", NO_FLAGS);
    defer expected_grad.release();
    std.testing.expect(array.equal(input.grad.?, expected_grad.data));
}

const ReduceArgs = struct {
    dims: array.DimArray,
    keepdims: bool,
    op: array.ReduceOperation,
};

fn reduceAlloc(alc: *std.mem.Allocator, x: Tensor, dims: []const u64, keepdims: bool, op: array.ReduceOperation) !Tensor {
    var args = try alc.create(ReduceArgs);
    args.* = ReduceArgs{
        .dims = array.DimArray.init(dims),
        .keepdims = keepdims,
        .op = op,
    };
    var result = autogradAlloc(alc, &[_]Tensor{x}, &[_]Array{x.data}, @ptrToInt(args), reduceForwardAlloc, reduceBackwardAlloc, reduceDealloc);
    return result;
}

fn reduceDealloc(alc: *std.mem.Allocator, extra_args_ptr: u64) void {
    var args = @intToPtr(*ReduceArgs, extra_args_ptr);
    alc.destroy(args);
}

fn reduceForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    var args = @intToPtr(*ReduceArgs, extra_args_ptr);
    return try array.reduceAlloc(alc, inputs[0], args.dims.getSlice(), args.keepdims, args.op);
}

fn reduceBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (inputs.len != 1) {
        @panic("Invalid number of inputs");
    }

    var args = @intToPtr(*ReduceArgs, extra_args_ptr);
    var reduced_numel: u64 = 1;
    var expand_shape = array.DimArray.init(inputs[0].getShape());
    // if we reduced along a dimension, set its size to 1 in the expanded shape of grad_output
    for (args.dims.getSlice()) |d| {
        reduced_numel *= expand_shape.array[d];
        expand_shape.array[d] = 1;
    }
    var expanded_grad_output = grad_output.reshapeView(expand_shape.getSlice());
    if (grad_inputs_out[0]) |grad| {
        switch (args.op) {
            .sum => array.copy(expanded_grad_output, grad),
            .max => {
                if (args.dims.ndim != 1) {
                    @panic("Too many dimensions for max");
                }
                var dim = args.dims.array[0];
                var index = try array.keepArgMaxAlloc(alc, inputs[0], dim);
                defer index.release();
                array.scatter(expanded_grad_output, grad, dim, index);
            },
            .mean => {
                array.copy(expanded_grad_output, grad);
                var divisor = try array.scalarAlloc(alc, grad.dtype, @intToFloat(f64, reduced_numel));
                defer divisor.release();
                array.divide(grad, divisor, grad);
            },
        }
    }
}

pub fn reduceSumAlloc(alc: *std.mem.Allocator, x: Tensor, dims: []const u64) !Tensor {
    return reduceAlloc(alc, x, dims, false, .sum);
}

pub fn keepSumAlloc(alc: *std.mem.Allocator, x: Tensor, dims: []const u64) !Tensor {
    return reduceAlloc(alc, x, dims, true, .sum);
}

pub fn reduceMaxAlloc(alc: *std.mem.Allocator, x: Tensor, dim: u64) !Tensor {
    return reduceAlloc(alc, x, &[_]u64{dim}, false, .max);
}

pub fn keepMaxAlloc(alc: *std.mem.Allocator, x: Tensor, dim: u64) !Tensor {
    return reduceAlloc(alc, x, &[_]u64{dim}, true, .max);
}

pub fn reduceMeanAlloc(alc: *std.mem.Allocator, x: Tensor, dims: []const u64) !Tensor {
    return reduceAlloc(alc, x, dims, false, .mean);
}

pub fn keepMeanAlloc(alc: *std.mem.Allocator, x: Tensor, dims: []const u64) !Tensor {
    return reduceAlloc(alc, x, dims, true, .mean);
}

test "reduce_sum_grad" {
    const a = try Tensor.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0, REQUIRES_GRAD);
    defer a.release();
    const b = try reduceSumAlloc(std.testing.allocator, a, &[_]u64{ 1, 2 });
    defer b.release();
    var c = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{2}, 3.0, 0);
    defer c.release();
    try backwardAlloc(std.testing.allocator, b, c);
    var d = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 3.0, 0);
    defer d.release();
    std.testing.expect(array.equal(a.grad.?, d.data));
}

test "keep_sum_grad" {
    const a = try Tensor.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0, REQUIRES_GRAD);
    defer a.release();
    const b = try keepSumAlloc(std.testing.allocator, a, &[_]u64{ 1, 2 });
    defer b.release();
    var c = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 1, 1 }, 3.0, 0);
    defer c.release();
    try backwardAlloc(std.testing.allocator, b, c);
    var d = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 3.0, 0);
    defer d.release();
    std.testing.expect(array.equal(a.grad.?, d.data));
}

test "reduce_mean_grad" {
    const input = try Tensor.allocWithString(f32, std.testing.allocator, "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", REQUIRES_GRAD);
    defer input.release();
    const output = try reduceMeanAlloc(std.testing.allocator, input, &[_]u64{1});
    defer output.release();
    var grad_output = try Tensor.allocWithValue(f32, std.testing.allocator, output.data.getShape(), 6.0, 0);
    defer grad_output.release();
    try backwardAlloc(std.testing.allocator, output, grad_output);
    const expected_grad_input = try Tensor.allocWithString(f32, std.testing.allocator, "[[2, 2, 2], [2, 2, 2], [2, 2, 2]]", NO_FLAGS);
    defer expected_grad_input.release();
    std.testing.expect(array.equal(input.grad.?, expected_grad_input.data));
}

test "reduce_max_grad" {
    const TestCase = struct {
        input: []const u8,
        dim: u64,
        expected_grad: []const u8,
    };
    var testcases = [_]TestCase{
        TestCase{
            .input = "[[[0,1], [1,0], [1,1]]]",
            .dim = 2,
            .expected_grad = "[[[0,3], [3,0], [3,0]]]",
        },
        TestCase{
            .input = "[[[0,1], [1,0], [1,1]]]",
            .dim = 1,
            .expected_grad = "[[[0,3], [3,0], [0,0]]]",
        },
    };

    for (testcases) |tc| {
        const input = try Tensor.allocWithString(f32, std.testing.allocator, tc.input, REQUIRES_GRAD);
        defer input.release();
        const output = try reduceMaxAlloc(std.testing.allocator, input, tc.dim);
        defer output.release();
        const grad_output = try Tensor.allocWithValue(f32, std.testing.allocator, output.data.getShape(), 3.0, 0);
        defer grad_output.release();
        try backwardAlloc(std.testing.allocator, output, grad_output);
        const expected_grad = try Tensor.allocWithString(f32, std.testing.allocator, tc.expected_grad, 0);
        defer expected_grad.release();
        std.testing.expect(array.equal(input.grad.?, expected_grad.data));
    }
}

pub fn reduceSumExprAlloc(alc: *std.mem.Allocator, x: Tensor, dims: Tensor) !Tensor {
    var dims_cast = try castAlloc(alc, dims, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.data.getBuffer(u64);
    return try reduceSumAlloc(alc, x, dims_buf);
}

pub fn keepSumExprAlloc(alc: *std.mem.Allocator, x: Tensor, dims: Tensor) !Tensor {
    var dims_cast = try castAlloc(alc, dims, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.data.getBuffer(u64);
    return try keepSumAlloc(alc, x, dims_buf);
}

pub fn reduceMaxExprAlloc(alc: *std.mem.Allocator, x: Tensor, dim: Tensor) !Tensor {
    var dims_cast = try castAlloc(alc, dim, .u64);
    defer dims_cast.release();
    return reduceMaxAlloc(alc, x, dims_cast.data.getItem(u64));
}

pub fn keepMaxExprAlloc(alc: *std.mem.Allocator, x: Tensor, dim: Tensor) !Tensor {
    var dims_cast = try castAlloc(alc, dim, .u64);
    defer dims_cast.release();
    return keepMaxAlloc(alc, x, dims_cast.data.getItem(u64));
}

pub fn reduceMeanExprAlloc(alc: *std.mem.Allocator, x: Tensor, dim: Tensor) !Tensor {
    var dims_cast = try castAlloc(alc, dim, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.data.getBuffer(u64);
    return reduceMeanAlloc(alc, x, dims_buf);
}

pub fn keepMeanExprAlloc(alc: *std.mem.Allocator, x: Tensor, dim: Tensor) !Tensor {
    var dims_cast = try castAlloc(alc, dim, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.data.getBuffer(u64);
    return keepMeanAlloc(alc, x, dims_buf);
}

pub fn gatherExprAlloc(alc: *std.mem.Allocator, x: Tensor, dim: Tensor, index: Tensor) !Tensor {
    var dims_cast = try castAlloc(alc, dim, .u64);
    defer dims_cast.release();
    var index_cast = try castAlloc(alc, index, .u64);
    defer index_cast.release();
    return gatherAlloc(alc, x, dims_cast.data.getItem(u64), index_cast);
}

pub fn transposeAlloc(alc: *std.mem.Allocator, x: Tensor) !Tensor {
    return autogradAlloc(alc, &[_]Tensor{x}, &[_]Array{x.data}, 0, transposeForwardAlloc, transposeBackwardAlloc, null);
}

pub fn transposeForwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64) ForwardError!Array {
    return array.transposeAlloc(alc, inputs[0]);
}

pub fn transposeBackwardAlloc(alc: *std.mem.Allocator, inputs: []Array, extra_args_ptr: u64, output: Array, grad_inputs_out: []?Array, grad_output: Array) BackwardError!void {
    if (inputs.len != 1) {
        @panic("invalid number of inputs");
    }

    if (grad_inputs_out[0]) |grad| {
        array.transpose(grad_output, grad);
    }
}

test "transpose_gradcheck" {
    var a = try Tensor.allocWithRange(f64, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0, REQUIRES_GRAD);
    defer a.release();
    var inputs = [_]Tensor{a};
    std.testing.expect(try gradCheck(f64, std.testing.allocator, transposeForwardAlloc, transposeBackwardAlloc, &inputs, 0));
}

pub fn autogradAlloc(alc: *std.mem.Allocator, inputs: []Tensor, input_arrays: []Array, extra_args_ptr: u64, forwardFn: ForwardFn, backwardFn: BackwardFn, deallocFn: ?DeallocFn) !Tensor {
    var output_array = try forwardFn(alc, input_arrays, extra_args_ptr);
    var requires_grad = false;
    for (inputs) |input| {
        if (input.requires_grad) {
            requires_grad = true;
        }
    }
    var flags: u64 = IS_BRANCH;
    if (requires_grad) {
        flags |= REQUIRES_GRAD;
    }
    var output = try Tensor.allocWithData(alc, output_array, flags);
    output_array.release();
    if (requires_grad) {
        var gr = try alc.create(GradientRecord);
        gr.* = try GradientRecord.alloc(alc, inputs, extra_args_ptr, output, backwardFn, deallocFn);
        output.grad_record = gr;
    } else {
        if (deallocFn) |dealloc| {
            dealloc(alc, extra_args_ptr);
        }
    }
    return output;
}

const RecordQueue = struct {
    const Queue = std.TailQueue(*GradientRecord);

    queue: Queue,
    alc: *std.mem.Allocator,

    const Self = @This();

    pub fn init(alc: *std.mem.Allocator) Self {
        return Self{ .queue = Queue{}, .alc = alc };
    }

    pub fn empty(self: *Self) bool {
        return self.queue.len == 0;
    }

    pub fn pushNode(self: *Self, grad_record: *GradientRecord) !void {
        var queue_node = try self.alc.create(Queue.Node);
        queue_node.data = grad_record;
        self.queue.append(queue_node);
    }

    pub fn popNode(self: *Self) *GradientRecord {
        var maybe_queue_node = self.queue.popFirst();
        if (maybe_queue_node) |queue_node| {
            var gr = queue_node.data;
            self.alc.destroy(queue_node);
            return gr;
        } else {
            @panic("attempted to dequeue from empty queue");
        }
    }
};

fn Counter(comptime T: type) type {
    return struct {
        map: std.AutoHashMap(T, u64),

        const Self = @This();

        fn init(alc: *std.mem.Allocator) !Self {
            return Self{ .map = std.AutoHashMap(T, u64).init(alc) };
        }

        fn incr(self: *Self, key: T) !u64 {
            var count: u64 = 1;
            if (self.map.get(key)) |c| {
                count += c;
            }
            try self.map.put(key, count);
            return count;
        }

        fn decr(self: *Self, key: T) !u64 {
            var count: u64 = 0;
            if (self.map.get(key)) |c| {
                count = c - 1;
            }
            try self.map.put(key, count);
            return count;
        }

        fn deinit(self: *Self) void {
            self.map.deinit();
        }
    };
}

fn toposort(alc: *std.mem.Allocator, root: *GradientRecord, records: []*GradientRecord) !void {
    var incoming_edge_counter = try Counter(*GradientRecord).init(std.testing.allocator);
    defer incoming_edge_counter.deinit();

    for (records) |rec| {
        for (rec.inputs) |input| {
            if (input.grad_record) |input_rec| {
                _ = try incoming_edge_counter.incr(input_rec);
            }
        }
    }

    var sorted_records = std.ArrayList(*GradientRecord).init(std.testing.allocator);
    defer sorted_records.deinit();
    var q = RecordQueue.init(std.testing.allocator);
    try q.pushNode(root);
    while (!q.empty()) {
        var rec = q.popNode();
        try sorted_records.append(rec);
        for (rec.inputs) |input| {
            if (input.grad_record) |input_rec| {
                var count = try incoming_edge_counter.decr(input_rec);
                if (count == 0) {
                    try q.pushNode(input_rec);
                }
            }
        }
    }
    if (sorted_records.items.len != records.len) {
        @panic("Failed to sort graph");
    }
    std.mem.copy(*GradientRecord, records, sorted_records.items);
}

pub fn backwardScalarAlloc(alc: *std.mem.Allocator, output: Tensor) !void {
    if (output.data.ndim != 0) {
        std.debug.panic("Expected scalar, got ndim {}", .{output.data.ndim});
    }
    var grad_output = try onesLikeAlloc(alc, output, NO_FLAGS);
    defer grad_output.release();
    return backwardAlloc(alc, output, grad_output);
}

pub fn backwardAlloc(alc: *std.mem.Allocator, output: Tensor, grad_output: Tensor) !void {
    if (output.grad_record == null) {
        return;
    }

    if (!std.mem.eql(u64, output.data.getShape(), grad_output.data.getShape())) {
        @panic("output shape does not match grad_output shape");
    }

    grad_output.data.retain();
    output.grad_record.?.grad_output = grad_output.data;
    var root = output.grad_record.?;

    // find all gradient records
    var seen = std.AutoHashMap(*GradientRecord, bool).init(std.testing.allocator);
    defer seen.deinit();
    var q = RecordQueue.init(std.testing.allocator);
    var records = std.ArrayList(*GradientRecord).init(std.testing.allocator);
    defer records.deinit();
    try q.pushNode(root);
    while (!q.empty()) {
        var rec = q.popNode();
        if (seen.get(rec) != null) {
            continue;
        }
        try records.append(rec);
        try seen.put(rec, true);
        for (rec.inputs) |input| {
            if (input.grad_record) |grad_record| {
                try q.pushNode(grad_record);
            }
        }
    }

    // sort the records
    try toposort(std.testing.allocator, root, records.items);

    // perform backward pass
    for (records.items) |rec| {
        var inputs = try alc.alloc(Array, rec.inputs.len);
        defer alc.free(inputs);
        var grad_inputs = try alc.alloc(?Array, rec.inputs.len);
        defer alc.free(grad_inputs);
        for (grad_inputs) |_, i| {
            if (rec.inputs[i].requires_grad) {
                grad_inputs[i] = try array.zerosLikeAlloc(alc, rec.inputs[i].data);
            } else {
                grad_inputs[i] = null;
            }
            inputs[i] = rec.inputs[i].data;
        }
        try rec.backward_fn(alc, inputs, rec.extra_args_ptr, rec.output.data, grad_inputs, rec.grad_output.?);
        rec.grad_output.?.release();
        rec.grad_output = null;
        for (grad_inputs) |maybe_grad_input, i| {
            if (maybe_grad_input == null) {
                continue;
            }

            var grad_input = maybe_grad_input.?;
            var input = rec.inputs[i];
            defer grad_input.release();

            if (!input.requires_grad) {
                @panic("Input does not require grad but we created a grad input for it");
            }

            if (input.is_leaf) {
                if (input.grad == null) {
                    @panic("missing grad buffer on leaf variable");
                }
                if (input.grad_record != null) {
                    @panic("leaf variable has grad record");
                }
            } else {
                if (input.grad_record == null) {
                    @panic("non-leaf tensor requires grad but has no grad record");
                }
            }

            if (input.grad) |input_grad| {
                array.plus(input_grad, grad_input, input_grad);
            }

            if (input.grad_record) |input_grad_record| {
                // enqueue a node to run backward on it
                // this node now owns the grad_input array
                if (input_grad_record.grad_output) |gradout| {
                    // there's already a grad output on this node, accumulate into it
                    array.plus(gradout, grad_input, gradout);
                } else {
                    // there's no grad output, put this value there
                    grad_input.retain();
                    input_grad_record.grad_output = grad_input;
                }
            }
        }
    }
}

fn gradCheck(comptime T: type, alc: *std.mem.Allocator, forwardFn: ForwardFn, backwardFn: BackwardFn, inputs: []Tensor, extra_args_ptr: u64) !bool {
    const epsilon = 1e-6;
    const rtol = 1e-3;
    const atol = 1e-5;

    for (inputs) |input| {
        if (!input.data.is_contiguous) {
            @panic("contiguous inputs required");
        }
    }
    var input_arrays = try alc.alloc(Array, inputs.len);
    defer alc.free(input_arrays);
    for (inputs) |input, i| {
        input_arrays[i] = input.data;
    }
    var output = try forwardFn(alc, input_arrays, extra_args_ptr);
    defer output.release();

    for (inputs) |input, input_index| {
        if (input.requires_grad) {
            var fd_jacobian = try Array.allocWithValue(T, alc, &[_]u64{ output.numel, input.data.numel }, 0.0);
            defer fd_jacobian.release();

            // use finite differences to build up the jacobian column by column
            var input_elem_index: u64 = 0;
            while (input_elem_index < input.data.numel) : (input_elem_index += 1) {
                var buf = input.data.getBuffer(T);
                var val = buf[input_elem_index];
                buf[input_elem_index] = val + epsilon;
                var plus_output = try forwardFn(alc, input_arrays, extra_args_ptr);
                defer plus_output.release();
                buf[input_elem_index] = val - epsilon;
                var minus_output = try forwardFn(alc, input_arrays, extra_args_ptr);
                defer minus_output.release();
                buf[input_elem_index] = val;
                var diff = try array.minusAlloc(alc, plus_output, minus_output);
                defer diff.release();
                var divisor = try Array.allocWithValue(T, alc, &[_]u64{}, 2.0 * epsilon);
                defer divisor.release();
                var jacobian_column = try array.divideAlloc(alc, diff, divisor);
                defer jacobian_column.release();
                var fd_column = fd_jacobian.narrowView(&[_]u64{ 0, input_elem_index }, &[_]u64{ output.numel, 1 });
                array.copy(jacobian_column.flatView(), fd_column.flatView());
            }

            // use our backward functions to build up the jacobian row by row
            var backward_jacobian = try Array.allocWithValue(T, alc, &[_]u64{ output.numel, input.data.numel }, 0.0);
            defer backward_jacobian.release();

            var output_elem_index: u64 = 0;
            while (output_elem_index < output.numel) : (output_elem_index += 1) {
                var grad_inputs = try alc.alloc(?Array, inputs.len);
                defer alc.free(grad_inputs);
                for (grad_inputs) |_, i| {
                    grad_inputs[i] = try array.zerosLikeAlloc(alc, input_arrays[i]);
                }
                var grad_output = try array.zerosLikeAlloc(alc, output);
                var buf = grad_output.getBuffer(T);
                buf[output_elem_index] = 1.0;
                defer grad_output.release();
                try backwardFn(alc, input_arrays, extra_args_ptr, output, grad_inputs, grad_output);
                var jacobian_row = grad_inputs[input_index].?;
                array.copy(jacobian_row.flatView(), backward_jacobian.narrowView(&[_]u64{ output_elem_index, 0 }, &[_]u64{ 1, input.data.numel }).flatView());
                for (grad_inputs) |maybe_grad_input| {
                    if (maybe_grad_input) |grad_input| {
                        grad_input.release();
                    }
                }
            }

            if (!array.allclose(fd_jacobian, backward_jacobian, rtol, atol)) {
                std.debug.print("jacobian mismatch\n", .{});
                std.debug.print("fd_jacobian {}\n", .{fd_jacobian});
                std.debug.print("backward_jacobian {}\n", .{backward_jacobian});
                return false;
            }
        }
    }
    return true;
}

test "plus_gradcheck" {
    var a = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 3 }, 1.0, REQUIRES_GRAD);
    defer a.release();
    var b = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 3 }, 2.0, REQUIRES_GRAD);
    defer b.release();
    var inputs = [_]Tensor{ a, b };
    std.testing.expect(try gradCheck(f64, std.testing.allocator, plusForwardAlloc, plusBackwardAlloc, &inputs, 0));
}

test "plus_grad" {
    var a = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 1.0, REQUIRES_GRAD);
    defer a.release();
    var b = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 2.0, REQUIRES_GRAD);
    defer b.release();
    var out = try plusAlloc(std.testing.allocator, a, b);
    defer out.release();
    var grad_out = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 4.0, 0);
    defer grad_out.release();
    try backwardAlloc(std.testing.allocator, out, grad_out);
    std.testing.expect(array.equal(a.grad.?, grad_out.data));
    std.testing.expect(array.equal(b.grad.?, grad_out.data));
}

test "plus_grad_multiple_levels" {
    var a = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 1.0, REQUIRES_GRAD);
    defer a.release();
    var b = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 2.0, REQUIRES_GRAD);
    defer b.release();
    var c = try plusAlloc(std.testing.allocator, a, b);
    defer c.release();
    var d = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 2.0, REQUIRES_GRAD);
    defer d.release();
    var e = try plusAlloc(std.testing.allocator, c, d);
    defer e.release();
    var f = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 3.0, 0);
    defer f.release();
    try backwardAlloc(std.testing.allocator, e, f);
    std.testing.expect(array.equal(a.grad.?, f.data));
    std.testing.expect(array.equal(b.grad.?, f.data));
}

test "plus_grad_bcast" {
    var a = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 1.0, REQUIRES_GRAD);
    defer a.release();
    var b = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{4}, 2.0, REQUIRES_GRAD);
    defer b.release();
    var c = try plusAlloc(std.testing.allocator, a, b);
    defer c.release();
    var d = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 3.0, 0);
    defer d.release();
    try backwardAlloc(std.testing.allocator, c, d);
    std.testing.expect(array.equal(a.grad.?, d.data));
    var e = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{4}, 18.0);
    defer e.release();
    std.testing.expect(array.equal(b.grad.?, e));
}

test "plus_no_grad" {
    var a = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 1.0, 0);
    defer a.release();
    var b = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 2.0, 0);
    defer b.release();
    var c = try plusAlloc(std.testing.allocator, a, b);
    defer c.release();
    var d = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 3.0, 0);
    defer d.release();
    try backwardAlloc(std.testing.allocator, c, d);
    std.testing.expect(a.grad == null);
    std.testing.expect(b.grad == null);
}

test "minus_gradcheck" {
    {
        var a = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 1, 3 }, 1.0, REQUIRES_GRAD);
        defer a.release();
        var b = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 3 }, 2.0, REQUIRES_GRAD);
        defer b.release();
        var inputs = [_]Tensor{ a, b };
        std.testing.expect(try gradCheck(f64, std.testing.allocator, minusForwardAlloc, minusBackwardAlloc, &inputs, 0));
    }

    {
        var a = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{3}, 1.0, REQUIRES_GRAD);
        defer a.release();
        var b = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 3 }, 2.0, REQUIRES_GRAD);
        defer b.release();
        var inputs = [_]Tensor{ a, b };
        std.testing.expect(try gradCheck(f64, std.testing.allocator, minusForwardAlloc, minusBackwardAlloc, &inputs, 0));
    }
}

test "times_gradcheck" {
    var a = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{3}, 2.0, REQUIRES_GRAD);
    defer a.release();
    var b = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 3 }, 3.0, REQUIRES_GRAD);
    defer b.release();
    var inputs = [_]Tensor{ a, b };
    std.testing.expect(try gradCheck(f64, std.testing.allocator, timesForwardAlloc, timesBackwardAlloc, &inputs, 0));
}

test "divide_gradcheck" {
    var a = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{3}, 2.0, REQUIRES_GRAD);
    defer a.release();
    var b = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 3 }, 3.0, REQUIRES_GRAD);
    defer b.release();
    var inputs = [_]Tensor{ a, b };
    std.testing.expect(try gradCheck(f64, std.testing.allocator, divideForwardAlloc, divideBackwardAlloc, &inputs, 0));
}

test "power_gradcheck" {
    var a = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{3}, 2.0, REQUIRES_GRAD);
    defer a.release();
    var b = try Tensor.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 3 }, 3.0, REQUIRES_GRAD);
    defer b.release();
    var inputs = [_]Tensor{ a, b };
    std.testing.expect(try gradCheck(f64, std.testing.allocator, powerForwardAlloc, powerBackwardAlloc, &inputs, 0));
}

fn getDType(t: Tensor) DType {
    return t.data.dtype;
}

pub fn binaryNotImplemented(alc: *std.mem.Allocator, x: Tensor, y: Tensor) !Tensor {
    @panic("operation not implemented");
}

pub fn unaryNotImplemented(alc: *std.mem.Allocator, x: Tensor) !Tensor {
    @panic("operation not implemented");
}

pub fn scalarNotImplemented(alc: *std.mem.Allocator, dtype: DType, value: f64) !Tensor {
    @panic("scalar not implemented");
}

pub fn expr(alc: *std.mem.Allocator, comptime exp: []const u8, args: anytype) !Tensor {
    comptime var opsTable = array.OpsTable(Tensor){
        .plus = plusAlloc,
        .minus = minusAlloc,
        .uplus = uplusAlloc,
        .uminus = uminusAlloc,
        .times = timesAlloc,
        .mtimes = mtimesAlloc,
        .divide = divideAlloc,
        .mdivide = binaryNotImplemented,
        .power = powerAlloc,
        .mpower = binaryNotImplemented,
        .eq = binaryNotImplemented,
        .gt = binaryNotImplemented,
        .gte = binaryNotImplemented,
        .lt = binaryNotImplemented,
        .lte = binaryNotImplemented,
        .transpose = transposeAlloc,
        .ctranspose = transposeAlloc,
        .scalar = scalarAlloc,
        .cast = castAlloc,
        .detach = detachAlloc,
        .log = logAlloc,
        .log2 = log2Alloc,
        .exp = expAlloc,
        .max = maxAlloc,
        .reduce_sum = reduceSumExprAlloc,
        .keep_sum = keepSumExprAlloc,
        .reduce_max = reduceMaxExprAlloc,
        .keep_max = keepMaxExprAlloc,
        .reduce_mean = reduceMeanExprAlloc,
        .keep_mean = keepMeanExprAlloc,
        .reduce_arg_max = binaryNotImplemented,
        .keep_arg_max = binaryNotImplemented,
        .gather = gatherExprAlloc,
        .get_dtype = getDType,
    };
    return try array.genericExpr(Tensor, opsTable, alc, exp, args);
}

test "expr" {
    {
        const a_data = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 1.0, 1.0);
        defer a_data.release();
        const b_data = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 1.0, 1.0);
        defer b_data.release();
        const a = try Tensor.allocWithData(std.testing.allocator, a_data, REQUIRES_GRAD);
        defer a.release();
        const b = try Tensor.allocWithData(std.testing.allocator, b_data, REQUIRES_GRAD);
        defer b.release();
        var c = try expr(std.testing.allocator, "a + b", .{ .a = a, .b = b });
        defer c.release();
        var d = try plusAlloc(std.testing.allocator, a, b);
        defer d.release();
        std.testing.expect(array.equal(c.data, d.data));
        var e = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 3.0, 0);
        defer e.release();
        try backwardAlloc(std.testing.allocator, c, e);
        std.testing.expect(array.equal(a.grad.?, e.data));
        std.testing.expect(array.equal(b.grad.?, e.data));
    }
}