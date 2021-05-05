// Combine parameters and some operations on them into reusable components
//
// These are just structs, the only thing special about them are they have a
// collectParameters method so that optimizers can find all of the parameters
// of nested modules.

const std = @import("std");
const array = @import("array.zig");
const Array = array.Array;
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const funcs = @import("funcs.zig");
const optim = @import("optim.zig");

pub const Parameter = struct {
    path: []const u8,
    value: Tensor,
};

pub const ParameterCollection = struct {
    data: []Parameter,
    count: u64,
    alc: *std.mem.Allocator,

    const Self = @This();

    fn init(alc: *std.mem.Allocator) !Self {
        var data = try alc.alloc(Parameter, 1);
        return Self{.data=data, .alc=alc, .count=0};
    }

    fn deinit(self: *Self) void {
        var i : usize = 0;
        while (i < self.count) : (i += 1) {
            self.alc.free(self.data[i].path);
        }
        self.alc.free(self.data);
    }

    fn append(self: *Self, path: []u8, value: Tensor) !void {
        if (self.data.len == self.count) {
            var new_data = try self.alc.alloc(Parameter, self.count*2);
            std.mem.copy(Parameter, new_data, self.data);
            self.alc.free(self.data);
            self.data = new_data;
        }
        var path_copy = try self.alc.alloc(u8, path.len);
        std.mem.copy(u8, path_copy, path);
        self.data[self.count] = Parameter{.path=path_copy, .value=value};
        self.count += 1;
    }
};

pub const ParameterCollector = struct {
    const Self = @This();

    prefix: []const u8,
    collection: *ParameterCollection,
    parent: ?*const Self,
    alc: *std.mem.Allocator,

    pub fn init(alc: *std.mem.Allocator) !Self {
        var collection = try alc.create(ParameterCollection);
        collection.* = try ParameterCollection.init(alc);
        return Self{.prefix="", .collection=collection, .alc=alc, .parent=null};
    }

    pub fn deinit(self: *Self) void {
        self.collection.deinit();
        self.alc.destroy(self.collection);
    }

    pub fn collectParameters(self: *const Self, obj: anytype, comptime name: []const u8) !void {
        try @field(obj, name).collectParameters(self.withPrefix(name));
    }

    pub fn collectSliceParameters(self: *const Self, obj: anytype, comptime name: []const u8) !void {
        var buf : [1024]u8 = undefined;
        var slice = @field(obj, name);
        for (slice) |item, index| {
            var prefix = try std.fmt.bufPrint(&buf, "{}[{}]", .{name, index});
            try item.collectParameters(self.withPrefix(prefix));
        }
    }

    pub fn addParameter(self: *const Self, obj: anytype, comptime name: []const u8) !void {
        var value = @field(obj, name);
        // traverse parent chain to build full path
        var cur : *const Self = self;
        var path_len : u64 = 0;
        while (cur.parent != null) {
            path_len += cur.prefix.len + 1;
            cur = cur.parent.?;
        }
        path_len += name.len;
        var path = try self.alc.alloc(u8, path_len);
        cur = self;
        var offset: u64 = path_len;
        offset -= name.len;
        std.mem.copy(u8, path[offset..], name);
        while (cur.parent != null) {
            offset -= 1;
            path[offset] = '.';
            offset -= cur.prefix.len;
            std.mem.copy(u8, path[offset..], cur.prefix);
            cur = cur.parent.?;
        }
        if (offset != 0) {
            @panic("Incorrect offset calculation");
        }
        try self.collection.append(path, value);
        self.alc.free(path);
    }

    pub fn getParameters(self: *const Self) []Parameter {
        return self.collection.data[0..self.collection.count];
    }

    pub fn withPrefix(self: *const Self, prefix: []const u8) Self {
        return Self{.prefix=prefix, .collection=self.collection, .alc=self.alc, .parent=self};
    }
};

pub const Dense = struct {
    weight: Tensor,
    bias: Tensor,

    const Self = @This();

    pub fn init(alc: *std.mem.Allocator, rng: *std.rand.Random, in_features: u64, out_features: u64) !Self {
        var weight = try tensor.zerosAlloc(alc, .f32, &[_]u64{in_features, out_features}, tensor.REQUIRES_GRAD);
        var bias = try tensor.zerosAlloc(alc, .f32, &[_]u64{out_features}, tensor.REQUIRES_GRAD);
        try funcs.kaimingUniform(alc, weight.data, rng);
        var high = try array.expr(alc, "1 ./ (in_features .^ 0.5)", .{.in_features=in_features});
        defer high.release();
        var low = try array.uminusAlloc(alc, high);
        defer low.release();
        array.fillUniform(bias.data, rng, low, high);
        return Self{.weight=weight, .bias=bias};
    }

    pub fn deinit(self: *Self) void {
        self.weight.release();
        self.bias.release();
    }

    pub fn collectParameters(self: Self, pc: ParameterCollector) !void {
        try pc.addParameter(self, "weight");
        try pc.addParameter(self, "bias");
    }

    pub fn forward(self: *Self, alc: *std.mem.Allocator, x: Tensor) !Tensor {
        return try tensor.expr(alc, "(x * weight) + bias", .{.x=x, .weight=self.weight, .bias=self.bias});
    }
};

pub const MLP = struct {
    fc1: Dense,
    fc2: Dense,

    const Self = @This();

    pub fn init(alc: *std.mem.Allocator, rng: *std.rand.Random, input_size: u64, hidden_size: u64, output_size: u64) !Self {
        return Self{.fc1=try Dense.init(alc, rng, input_size, hidden_size), .fc2=try Dense.init(alc, rng, hidden_size, output_size)};
    }

    pub fn deinit(self: *Self) void {
        self.fc1.deinit();
        self.fc2.deinit();
    }

    pub fn collectParameters(self: Self, pc: ParameterCollector) !void {
        try pc.collectParameters(self, "fc1");
        try pc.collectParameters(self, "fc2");
    }

    pub fn forward(self: *Self, alc: *std.mem.Allocator, x: Tensor) !Tensor {
        var fc1_out = try self.fc1.forward(alc, x);
        var fc1_act : Tensor = undefined;
        {
            defer fc1_out.release();
            fc1_act = try funcs.relu(alc, fc1_out);
        }
        var fc2_out : Tensor = undefined;
        {
            defer fc1_act.release();
            return try self.fc2.forward(alc, fc1_act);
        }
    }
};

test "mlp" {
    var in_features : u64 = 5;
    var hidden_features : u64 = 2;
    var out_features : u64 = 2;
    var gen = std.rand.Xoroshiro128.init(0);
    var mlp = try MLP.init(std.testing.allocator, &gen.random, in_features, hidden_features, out_features);
    defer mlp.deinit();

    mlp.fc1.weight.release();
    mlp.fc1.weight = try Tensor.allocWithRange(f32, std.testing.allocator, &[_]u64{in_features, hidden_features}, 0.0, 1.0, tensor.REQUIRES_GRAD);
    mlp.fc1.bias.release();
    mlp.fc1.bias = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{hidden_features}, 0, tensor.REQUIRES_GRAD);

    mlp.fc2.weight.release();
    mlp.fc2.weight = try Tensor.allocWithRange(f32, std.testing.allocator, &[_]u64{hidden_features, out_features}, 0.0, 1.0, tensor.REQUIRES_GRAD);
    mlp.fc2.bias.release();
    mlp.fc2.bias = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{out_features}, 0, tensor.REQUIRES_GRAD);

    var input = try Tensor.allocWithValue(f32, std.testing.allocator, &[_]u64{4, 5}, 1.0, tensor.NO_FLAGS);
    defer input.release();
    var target = try Tensor.allocWithValue(u64, std.testing.allocator, &[_]u64{4}, 0, tensor.NO_FLAGS);
    defer target.release();
    var logits = try mlp.forward(std.testing.allocator, input);
    defer logits.release();
    var output = try funcs.logSoftmax(std.testing.allocator, logits, &[_]u64{1});
    defer output.release();
    std.testing.expect(output.data.get(f32, &[_]u64{0,0}) == -45.0);
    std.testing.expect(output.data.get(f32, &[_]u64{0,1}) == 0.0);
    var loss = try funcs.nllLoss(std.testing.allocator, output, target);
    defer loss.release();
    var grad_output = try tensor.onesLikeAlloc(std.testing.allocator, loss, tensor.NO_FLAGS);
    defer grad_output.release();
    var pc = try ParameterCollector.init(std.testing.allocator);
    defer pc.deinit();
    try mlp.collectParameters(pc);
    var opt = try optim.SGD.init(std.testing.allocator, pc.getParameters(), 0.0);
    defer opt.deinit();
    try opt.zeroGrad();
    try tensor.backwardAlloc(std.testing.allocator, loss, grad_output);
    std.testing.expect(mlp.fc1.weight.grad.?.get(f32, &[_]u64{0,0}) == 1.0);
    std.testing.expect(mlp.fc2.weight.grad.?.get(f32, &[_]u64{0,0}) == -20.0);
    var before = mlp.fc2.weight.data.get(f32, &[_]u64{0,0});
    try opt.step(2);
    var after = mlp.fc2.weight.data.get(f32, &[_]u64{0,0});
    std.testing.expect(after - before == 40.0);
}