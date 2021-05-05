const std = @import("std");
const array = @import("array.zig");
const Array = array.Array;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const expr = tensor.expr;

pub fn relu(alc: *std.mem.Allocator, x: Tensor) !Tensor {
    return try expr(alc, "max(0, x)", .{ .x = x });
}

pub fn logSoftmax(alc: *std.mem.Allocator, x: Tensor, dims: []const u64) !Tensor {
    var dims_tensor = Tensor.flatFromBuffer(u64, @bitCast([]u64, dims));
    // https://github.com/google/jax/pull/2260
    var x_shifted = try expr(alc, "x - detach(keep_max(x, dims))", .{ .x = x, .dims = dims_tensor });
    defer x_shifted.release();
    return try expr(alc, "x_shifted - log(keep_sum(exp(x_shifted), dims))", .{ .x_shifted = x_shifted, .dims = dims_tensor });
}

test "logSoftmax" {
    var input = try Tensor.allocWithString(f32, std.testing.allocator, "[[1, 2, 3], [1, 20, 300], [0, -1, 1000], [0, -1, 1000]]", tensor.REQUIRES_GRAD);
    defer input.release();
    var dims = [_]u64{1};
    var output = try logSoftmax(std.testing.allocator, input, &dims);
    defer output.release();
    var expected_output = try Array.allocWithString(f32, std.testing.allocator, "[[-2.4076e+00, -1.4076e+00, -4.0761e-01], [-2.9900e+02, -2.8000e+02, 0.0000e+00], [-1.0000e+03, -1.0010e+03, 0.0000e+00], [-1.0000e+03, -1.0010e+03, 0.0000e+00]]");
    defer expected_output.release();
    std.testing.expect(array.allclose(output.data, expected_output, 1e-5, 1e-8));
    var grad_output = try tensor.onesLikeAlloc(std.testing.allocator, output, tensor.NO_FLAGS);
    defer grad_output.release();
    try tensor.backwardAlloc(std.testing.allocator, output, grad_output);
    var expected_grad_input = try Array.allocWithString(f32, std.testing.allocator, "[[ 0.7299,  0.2658, -0.9957], [ 1.0000,  1.0000, -2.0000], [1.0000,  1.0000, -2.0000], [1.0000,  1.0000, -2.0000]]");
    defer expected_grad_input.release();
    std.testing.expect(array.allclose(input.grad.?, expected_grad_input, 1e-5, 1e-3));
}

pub fn nllLoss(alc: *std.mem.Allocator, input: Tensor, target: Tensor) !Tensor {
    if (input.data.ndim != 2) {
        @panic("Input has wrong number of dimensions");
    }
    if (target.data.ndim != 1) {
        @panic("Target has wrong number of dimensions");
    }
    if (!array.dtypeIsInteger(target.data.dtype)) {
        @panic("Target dtype must be int");
    }
    var target_expanded = target.reshapeView(&[_]u64{target.data.numel, 1});
    var dims = [_]u64{0,1};
    return try expr(alc, "reduce_mean(-gather(input, 1, target), dims)", .{.input=input, .target=target_expanded, .dims=Tensor.flatFromBuffer(u64, &dims)});
}

test "nllLoss" {
    var input = try Tensor.allocWithString(f32, std.testing.allocator, "[[1, 2, 3], [1, 20, 300], [0, 1, 1000]]", tensor.REQUIRES_GRAD);
    defer input.release();
    var target = try Tensor.allocWithString(u64, std.testing.allocator, "[0, 1, 2]", tensor.NO_FLAGS);
    defer target.release();
    var output = try nllLoss(std.testing.allocator, input, target);
    defer output.release();
    var expected_output = try array.scalarAlloc(std.testing.allocator, .f32, -340.3333);
    defer expected_output.release();
    std.testing.expect(array.allclose(output.data, expected_output, 1e-05, 1e-08));
}

/// Kaiming init for fan_in init with gain set for ReLU
/// https://arxiv.org/abs/1502.01852
pub fn kaimingUniform(alc: *std.mem.Allocator, dst: Array, r: *std.rand.Random) !void {
    var high = try array.expr(alc, "(3.0 .^ 0.5) .* ((2.0 .^ 0.5) ./ (fan .^ 0.5))", .{.fan=dst.getShape()[0]});
    defer high.release();
    var low = try array.uminusAlloc(alc, high);
    defer low.release();
    array.fillUniform(dst, r, low, high);
}