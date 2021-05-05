// Optimizer implementations, each takes a list of parameters, and has
// zeroGrad and step methods.

const std = @import("std");
const array = @import("array.zig");
const Array = array.Array;
const module = @import("module.zig");

pub const SGD = struct {
    parameters: []module.Parameter,
    momentums: []Array,
    momentum: f32,
    alc: *std.mem.Allocator,

    const Self = @This();

    pub fn init(alc: *std.mem.Allocator, parameters: []module.Parameter, momentum: f32) !Self {
        var momentums = try alc.alloc(Array, parameters.len);
        for (momentums) |_, index| {
            momentums[index] = try array.zerosLikeAlloc(alc, parameters[index].value.grad.?);
        }
        return Self{.parameters=parameters, .alc=alc, .momentum=momentum, .momentums=momentums};
    }

    pub fn deinit(self: *Self) void {
        for (self.momentums) |v| {
            v.release();
        }
        self.alc.free(self.momentums);
    }

    pub fn zeroGrad(self: *Self) !void {
        for (self.parameters) |param| {
            var grad = param.value.grad.?;
            var zero = try array.scalarAlloc(self.alc, grad.dtype, 0.0);
            defer zero.release();
            array.copy(zero, grad);
        }
    }

    pub fn step(self: *Self, lr: f32) !void {
        for (self.parameters) |param, index| {
            var grad = param.value.grad.?;

            if (self.momentum != 0) {
                // update momentum
                var v = self.momentums[index];
                var new_v = try array.expr(self.alc, "m .* v .+ g", .{.m=self.momentum, .v=v, .g=grad});
                defer new_v.release();
                array.copy(new_v, v);
                grad = v;
            }

            var update = try array.expr(self.alc, "-lr .* g", .{.lr=lr, .g=grad});
            defer update.release();
            var data = param.value.data;
            array.plus(data, update, data);
        }
    }
};

// Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980
pub const Adam = struct {
    parameters: []module.Parameter,
    first_moments: []Array,
    second_moments: []Array,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step_count: u64,
    alc: *std.mem.Allocator,

    const Self = @This();

    pub fn init(alc: *std.mem.Allocator, parameters: []module.Parameter, beta1: f32, beta2: f32, epsilon: f32) !Self {
        var first_moments = try alc.alloc(Array, parameters.len);
        var second_moments = try alc.alloc(Array, parameters.len);
        for (first_moments) |_, index| {
            first_moments[index] = try array.zerosLikeAlloc(alc, parameters[index].value.grad.?);
            second_moments[index] = try array.zerosLikeAlloc(alc, parameters[index].value.grad.?);
        }
        return Self{.parameters=parameters, .alc=alc, .beta1=beta1, .beta2=beta2, .epsilon=epsilon, .step_count=0, .first_moments=first_moments, .second_moments=second_moments};
    }

    pub fn deinit(self: *Self) void {
        for (self.first_moments) |v| {
            v.release();
        }
        self.alc.free(self.first_moments);
        for (self.second_moments) |v| {
            v.release();
        }
        self.alc.free(self.second_moments);
    }

    pub fn zeroGrad(self: *Self) !void {
        for (self.parameters) |param| {
            var grad = param.value.grad.?;
            var zero = try array.scalarAlloc(self.alc, grad.dtype, 0.0);
            defer zero.release();
            array.copy(zero, grad);
        }
    }

    pub fn step(self: *Self, lr: f32) !void {
        self.step_count += 1;
        for (self.parameters) |param, index| {
            var grad = param.value.grad.?;

            var m = self.first_moments[index];
            var new_m = try array.expr(self.alc, "beta1 .* m + (1 - beta1) .* g", .{.m=m, .beta1=self.beta1, .g=grad});
            defer new_m.release();
            array.copy(new_m, m);

            var v = self.second_moments[index];
            var new_v = try array.expr(self.alc, "beta2 .* v + (1 - beta2) .* (g .* g)", .{.v=v, .beta2=self.beta2, .g=grad});
            defer new_v.release();
            array.copy(new_v, v);

            var m_hat = try array.expr(self.alc, "m ./ (1 - beta1 .^ step)", .{.m=m, .beta1=self.beta1, .step=self.step_count});
            defer m_hat.release();

            var v_hat = try array.expr(self.alc, "v ./ (1 - beta2 .^ step)", .{.v=v, .beta2=self.beta2, .step=self.step_count});
            defer v_hat.release();

            var update = try array.expr(self.alc, "-lr .* m_hat ./ (v_hat .^ 0.5 + epsilon)", .{.lr=lr, .m_hat=m_hat, .v_hat=v_hat, .epsilon=self.epsilon});
            defer update.release();
            var data = param.value.data;
            array.plus(data, update, data);
        }
    }
};