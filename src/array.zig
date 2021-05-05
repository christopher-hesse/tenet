// An N-dimensional array struct
//   The metadata is stored in the struct itself, and for this reason
//   there is a maximum number of dimensions, currently set to 8.
//   The size of this struct is maybe a bit large, the 
//
//   The buffer backing the array is allocated, and there is a reference
//   count to determine when to free it.  Arrays start with 1 reference
//   which is increased by .retain() and decreased by .release().
//
//   Here's the allocated data:
//      buffer: a slice that is used to store the data for the array
//      ref_counter: keeps track of the reference count for the array
//
//   Both of these are freed when the reference count hits zero.

const USE_MKL = @import("build_options").USE_MKL;

const std = @import("std");
const reference_counter = @import("reference_counter.zig");
const ReferenceCounter = reference_counter.ReferenceCounter;

const mkl = @import("mkl.zig");

// CUDNN_DIM_MAX is apparently 8
const maxNumDim = 8;
const defaultIntDType = DType.i64;
const defaultFloatDType = DType.f32;

pub fn sliceProduct(comptime T: type, arr: []const T) T {
    var result: T = 1;
    for (arr) |v| {
        result *= v;
    }
    return result;
}

fn indent(writer: anytype, level: u64) !void {
    var i: u64 = 0;
    while (i < level) : (i += 1) {
        try writer.writeAll(" ");
    }
}

fn contains(comptime T: type, haystack: []const T, needle: T) bool {
    for (haystack) |item| {
        if (needle == item) {
            return true;
        }
    }
    return false;
}

const PositionIterator = struct {
    remaining: u64,
    length: u64,
    ndim: u64,
    shape: [maxNumDim]u64 = [_]u64{0} ** maxNumDim,
    pos: [maxNumDim]u64 = [_]u64{0} ** maxNumDim,

    const Self = @This();

    fn init(shape: []const u64) Self {
        var length = sliceProduct(u64, shape);
        var inst = Self{ .length = length, .remaining = length, .ndim = shape.len };
        std.mem.copy(u64, &inst.shape, shape);
        return inst;
    }

    fn next(self: *Self) ?[]const u64 {
        if (self.remaining == 0) {
            return null;
        }

        self.remaining -= 1;

        if (self.remaining == self.length - 1) {
            // don't increase offset on first item
            return self.pos[0..self.ndim];
        }

        if (self.ndim == 0) {
            return null;
        }

        // increment our pos, moving offset according to strides
        // start at right side and move to the left
        var d: u64 = self.ndim - 1;
        self.pos[d] += 1;

        while (self.pos[d] == self.shape[d]) {
            std.debug.assert(d > 0);
            self.pos[d] = 0;
            self.pos[d - 1] += 1;
            d -= 1;
        }
        return self.pos[0..self.ndim];
    }
};

const StridedIterator = struct {
    remaining: u64,
    length: u64,
    offset: u64,
    ndim: u64,
    strides: [maxNumDim]u64 = [_]u64{0} ** maxNumDim,
    shape: [maxNumDim]u64 = [_]u64{0} ** maxNumDim,
    pos: [maxNumDim]u64 = [_]u64{0} ** maxNumDim,

    const Self = @This();

    fn init(shape: []const u64, strides: []const u64, offset: u64) Self {
        var length = sliceProduct(u64, shape);
        var inst = Self{ .length = length, .remaining = length, .offset = offset, .ndim = shape.len };
        std.mem.copy(u64, &inst.strides, strides);
        std.mem.copy(u64, &inst.shape, shape);
        return inst;
    }

    fn next(self: *Self) ?u64 {
        if (self.remaining == 0) {
            return null;
        }

        self.remaining -= 1;

        if (self.remaining == self.length - 1) {
            // don't increase offset on first item
            return self.offset;
        }

        // it seems this has to be its own if statement so that the compiler can realize that
        // the code below that depends on ndim > 0 is never executed
        if (self.ndim == 0) {
            return null;
        }

        // increment our pos, moving offset according to strides
        // start at right side and move to the left
        var d: u64 = self.ndim - 1;
        self.pos[d] += 1;
        self.offset += self.strides[d];

        while (self.pos[d] == self.shape[d]) {
            std.debug.assert(d > 0);
            self.pos[d] = 0;
            self.offset -= self.strides[d] * self.shape[d];
            self.pos[d - 1] += 1;
            self.offset += self.strides[d - 1];
            d -= 1;
        }
        return self.offset;
    }
};

pub const DType = enum {
    u8, // useful for pixel data
    u64, // useful for shapes
    i64,
    f32,
    f64, // useful for grad check
};

const DTypeBuffer = union(DType) {
    u8: []u8,
    u64: []u64,
    i64: []i64,
    f32: []f32,
    f64: []f64,
};

const DTypeValue = union(DType) {
    u8: u8,
    u64: u64,
    i64: i64,
    f32: f32,
    f64: f64,
};

fn typeToDType(comptime T: type) DType {
    return switch (T) {
        u8 => DType.u8,
        u64 => DType.u64,
        i64 => DType.i64,
        f32 => DType.f32,
        f64 => DType.f64,
        else => @compileError("unsupported type"),
    };
}

fn dtypeToTypeName(dtype: DType) []const u8 {
    return switch (dtype) {
        .u8 => "u8",
        .u64 => "u64",
        .i64 => "i64",
        .f32 => "f32",
        .f64 => "f64",
    };
}

fn dtypeToPriority(dtype: DType) u8 {
    return switch (dtype) {
        .u8 => 0,
        .u64 => 1,
        .i64 => 2,
        .f32 => 3,
        .f64 => 4,
    };
}

pub fn dtypeIsInteger(dtype: DType) bool {
    return switch (dtype) {
        .u8, .i64, .u64 => true,
        else => false,
    };
}

fn dtypeMax(dtype1: DType, dtype2: DType) DType {
    if (dtypeToPriority(dtype1) > dtypeToPriority(dtype2)) {
        return dtype1;
    } else {
        return dtype2;
    }
}

fn getShapeFromString(str: []const u8) DimArray {
    var ndim: u64 = 0;
    for (str) |c| {
        if (c == '[') {
            ndim += 1;
        } else if (c == ' ' or c == '\n') {
            continue;
        } else {
            break;
        }
    }
    var shape = DimArray{ .ndim = ndim };
    var dim_complete = [_]bool{false} ** maxNumDim;
    // assume we have at least 1 element per array
    for (dim_complete) |_, d| {
        shape.array[d] = 1;
    }
    var started: bool = false;
    var d: u64 = 0;
    for (str) |c| {
        if (c == '[') {
            if (started) {
                d += 1;
            } else {
                started = true;
            }
        } else if (c == ']') {
            dim_complete[d] = true;
            if (d == 0) {
                break;
            }
            d -= 1;
        } else if (c == ',' and !dim_complete[d]) {
            shape.array[d] += 1;
        }
    }
    return shape;
}

test "get_shape_from_string" {
    var shape = getShapeFromString("[[1,2], [3,4], [5,6]]");
    std.testing.expect(std.mem.eql(u64, shape.getSlice(), &[_]u64{ 3, 2 }));
    var shape2 = getShapeFromString("[[[1,2], [3,4], [5,6]], [[1,2], [3,4], [5,6]], [[1,2], [3,4], [5,6]], [[1,2], [3,4], [5,6]]]");
    std.testing.expect(std.mem.eql(u64, shape2.getSlice(), &[_]u64{ 4, 3, 2 }));
}

fn readBufferFromString(comptime T: type, buf: []T, str: []const u8) void {
    var buf_index: u64 = 0;
    var index: u64 = 0;
    while (index < str.len) {
        var c = str[index];
        if (('0' <= c and c <= '9') or c == '-') {
            var start = index;
            var end = index;
            while (('0' <= c and c <= '9') or c == '-' or c == '.' or c == 'e' or c == 'E' or c == '+') {
                end += 1;
                c = str[end];
            }
            var val = std.fmt.parseFloat(f64, str[start..end]) catch @panic("Failed to parse float");
            buf[buf_index] = switch (@typeInfo(T)) {
                .Int => @floatToInt(T, val),
                .Float => @floatCast(T, val),
                else => @panic("Unexpected type"),
            };
            buf_index += 1;
            index = end;
        } else {
            index += 1;
        }
    }
    if (buf_index != buf.len) {
        @panic("Values do not match expected shape");
    }
}

test "read_buffer_from_string" {
    var buf = [_]f32{0.0} ** 6;
    readBufferFromString(f32, &buf, "[[1,2], [3,4], [5,6]]");
    std.testing.expect(std.mem.eql(f32, &buf, &[_]f32{ 1, 2, 3, 4, 5, 6 }));
}

pub const Array = struct {
    buffer_union: DTypeBuffer,
    shape: DimArray,
    strides: DimArray,
    numel: u64,
    ndim: u64,
    dtype: DType,
    // it is nice to have multiple views onto the same underlying data
    // and not have to copy it each time, so keep a reference count
    // if we don't own the memory for the data slice, this will be null
    ref_counter: ?*ReferenceCounter,
    alc: ?*std.mem.Allocator,
    is_contiguous: bool,
    offset: u64,

    const Self = @This();

    fn calculateStrides(shape: []const u64, strides_out: []u64) void {
        var stride: u64 = 1;
        for (shape) |_, i| {
            strides_out[shape.len - 1 - i] = stride;
            stride *= shape[shape.len - 1 - i];
        }
    }

    fn createStrides(shape: []const u64) DimArray {
        var strides = DimArray{ .ndim = shape.len };
        calculateStrides(shape, strides.getSlice());
        return strides;
    }

    pub fn alloc(comptime T: type, alc: *std.mem.Allocator, shape: []const u64) !Self {
        var ndim = shape.len;
        var numel = sliceProduct(u64, shape);
        var ref_counter = try alc.create(ReferenceCounter);
        ref_counter.* = ReferenceCounter.init();
        var buf = try alc.alloc(T, numel);
        var buffer_union = switch (T) {
            u8 => DTypeBuffer{ .u8 = buf },
            u64 => DTypeBuffer{ .u64 = buf },
            i64 => DTypeBuffer{ .i64 = buf },
            f32 => DTypeBuffer{ .f32 = buf },
            f64 => DTypeBuffer{ .f64 = buf },
            else => @panic("invalid type"),
        };
        return Self{ .ndim = ndim, .numel = numel, .dtype = typeToDType(T), .ref_counter = ref_counter, .buffer_union = buffer_union, .alc = alc, .is_contiguous = true, .offset = 0, .shape = DimArray.init(shape), .strides = createStrides(shape) };
    }

    pub fn allocWithRange(comptime T: type, alc: *std.mem.Allocator, shape: []const u64, start: T, step: T) !Self {
        var t = try Self.alloc(T, alc, shape);
        var v = start;
        var buf = t.getBuffer(T);
        for (buf) |_, i| {
            buf[i] = v;
            v += step;
        }
        return t;
    }

    pub fn allocWithString(comptime T: type, alc: *std.mem.Allocator, str: []const u8) !Self {
        var shape = getShapeFromString(str);
        var t = try Self.allocWithValue(T, alc, shape.getSlice(), 0);
        readBufferFromString(T, t.getBuffer(T), str);
        return t;
    }

    pub fn allocWithValue(comptime T: type, alc: *std.mem.Allocator, shape: []const u64, value: T) !Self {
        var t = try Self.alloc(T, alc, shape);
        var buf = t.getBuffer(T);
        std.mem.set(T, buf, value);
        return t;
    }

    pub fn fromBuffer(comptime T: type, shape: []const u64, buf: []T) Self {
        var ndim = shape.len;
        var numel = sliceProduct(u64, shape);
        if (buf.len != numel) {
            @panic("data length does not match shape");
        }
        var data = switch (T) {
            u8 => DTypeBuffer{ .u8 = buf },
            u64 => DTypeBuffer{ .u64 = buf },
            i64 => DTypeBuffer{ .i64 = buf },
            f32 => DTypeBuffer{ .f32 = buf },
            f64 => DTypeBuffer{ .f64 = buf },
            else => @panic("invalid type"),
        };
        return Self{ .ndim = ndim, .numel = numel, .dtype = typeToDType(T), .ref_counter = null, .buffer_union = data, .alc = null, .is_contiguous = true, .offset = 0, .shape = DimArray.init(shape), .strides = createStrides(shape) };
    }

    pub fn flatFromBuffer(comptime T: type, buf: []T) Self {
        return Self.fromBuffer(T, &[_]u64{buf.len}, buf);
    }

    pub fn scalarFromBuffer(comptime T: type, buf: []T) Self {
        if (buf.len != 1) {
            std.debug.panic("Buffer length {} invalid for scalar, must be 1", .{buf.len});
        }
        return Self.fromBuffer(T, &[_]u64{}, buf);
    }

    pub fn getShape(self: *const Self) []const u64 {
        return self.shape.getConstSlice();
    }

    pub fn getStrides(self: *const Self) []const u64 {
        return self.strides.getConstSlice();
    }

    pub fn getBuffer(self: Self, comptime T: type) []T {
        return switch (T) {
            u8 => self.buffer_union.u8,
            u64 => self.buffer_union.u64,
            i64 => self.buffer_union.i64,
            f32 => self.buffer_union.f32,
            f64 => self.buffer_union.f64,
            else => @panic("invalid type"),
        };
    }

    pub fn flatView(self: Self) Self {
        if (self.ndim == 0) {
            @panic("attempted to flatten a scalar");
        }
        var cur_strides = self.getStrides();
        var stride: u64 = cur_strides[self.ndim - 1];
        if (!self.is_contiguous) {
            var non_singleton_dims: u64 = 0;
            for (self.getShape()) |s, i| {
                if (s != 1) {
                    non_singleton_dims += 1;
                    stride = cur_strides[i];
                }
            }
            if (non_singleton_dims > 1) {
                @panic("can only flatten contiguous tensors or tensors with one non-singleton dimenson");
            }
        }
        var inst = self;
        inst.ndim = 1;
        inst.shape = DimArray.init(&[_]u64{self.numel});
        inst.strides = DimArray.init(&[_]u64{stride});
        return inst;
    }

    pub fn expandView(self: Self, shape: []const u64) Self {
        if (shape.len < self.ndim) {
            @panic("new shape must have the same number or more dimensions");
        }
        // shift strides over to account for new dimensions, which are added on the left
        var strides = [_]u64{0} ** maxNumDim;
        const num_added_dims = shape.len - self.ndim;
        std.mem.copy(u64, strides[num_added_dims .. num_added_dims + self.ndim], self.getStrides());
        var contiguous = num_added_dims == 0;
        var output_index: u64 = 0;
        var cur_shape = self.getShape();
        while (output_index < num_added_dims) {
            // this is a newly added dimension, the stride will be 0
            strides[output_index] = 0;
            output_index += 1;
        }
        while (output_index < shape.len) {
            var input_index: u64 = output_index - num_added_dims;

            if (cur_shape[input_index] == 1) {
                if (shape[output_index] == 0) {
                    @panic("attempted to expand size 0 dimension");
                }
                // we can now iterate along this dimension without advancing through the data
                strides[output_index] = 0;
                contiguous = false;
            } else if (shape[output_index] != cur_shape[input_index]) {
                @panic("expanded shape not compatible with existing shape");
            }
            output_index += 1;
        }
        var numel = sliceProduct(u64, shape);
        return Self{ .ndim = shape.len, .numel = numel, .dtype = self.dtype, .ref_counter = self.ref_counter, .buffer_union = self.buffer_union, .alc = self.alc, .is_contiguous = contiguous, .offset = self.offset, .shape = DimArray.init(shape), .strides = DimArray.init(strides[0..shape.len]) };
    }

    pub fn narrowView(self: Self, pos: []const u64, shape: []const u64) Self {
        if (pos.len != self.ndim) {
            @panic("position has wrong number of dimensions");
        }
        if (shape.len != self.ndim) {
            @panic("shape has wrong number of dimensions");
        }
        var offset = self.offset;
        var d: u64 = 0;
        var is_contiguous = self.is_contiguous;
        var contiguous_strides = [_]u64{0} ** maxNumDim;
        calculateStrides(shape, &contiguous_strides);
        var cur_shape = self.getShape();
        var cur_strides = self.getStrides();
        while (d < self.ndim) : (d += 1) {
            offset += pos[d] * cur_strides[d];
            if (pos[d] + shape[d] > cur_shape[d]) {
                @panic("Position with shape exceeds size of source shape");
            }
            if (pos[d] >= cur_shape[d]) {
                @panic("Invalid position");
            }
            if (shape[d] != 1 and contiguous_strides[d] != cur_strides[d]) {
                is_contiguous = false;
            }
        }
        var numel = sliceProduct(u64, shape);
        return Self{ .ndim = self.ndim, .numel = numel, .dtype = self.dtype, .ref_counter = self.ref_counter, .buffer_union = self.buffer_union, .alc = self.alc, .is_contiguous = is_contiguous, .offset = offset, .shape = DimArray.init(shape), .strides = self.strides };
    }

    pub fn reshapeView(self: Self, shape: []const u64) Self {
        if (self.numel != sliceProduct(u64, shape)) {
            @panic("Attempted to reshape to different number of elements");
        }

        if (!self.is_contiguous) {
            @panic("Reshape view of non-contiguous arrays not yet supported");
        }
        return Self{ .ndim = shape.len, .numel = self.numel, .dtype = self.dtype, .ref_counter = self.ref_counter, .buffer_union = self.buffer_union, .alc = self.alc, .is_contiguous = self.is_contiguous, .offset = self.offset, .shape = DimArray.init(shape), .strides = createStrides(shape) };
    }

    pub fn createIterator(self: Self) StridedIterator {
        return StridedIterator.init(self.getShape(), self.getStrides(), self.offset);
    }

    pub fn get(self: Self, comptime T: type, pos: []const u64) T {
        var offset: u64 = self.offset;
        if (pos.len != self.ndim) {
            @panic("position has wrong number of dimensions");
        }
        for (self.getStrides()) |stride, i| {
            offset += pos[i] * stride;
        }
        var buf = self.getBuffer(T);
        return buf[offset];
    }

    pub fn getValue(self: Self, pos: []const u64) DTypeValue {
        return switch (self.dtype) {
            .u8 => DTypeValue{ .u8 = self.get(u8, pos) },
            .u64 => DTypeValue{ .u64 = self.get(u64, pos) },
            .i64 => DTypeValue{ .i64 = self.get(i64, pos) },
            .f32 => DTypeValue{ .f32 = self.get(f32, pos) },
            .f64 => DTypeValue{ .f64 = self.get(f64, pos) },
        };
    }

    pub fn getItem(self: Self, comptime T: type) T {
        if (self.numel != 1) {
            @panic("Can only call getItem on single-element Tensors");
        }
        return self.getBuffer(T)[self.offset];
    }

    pub fn set(self: Self, comptime T: type, pos: []const u64, value: T) void {
        var offset: u64 = self.offset;
        for (self.getStrides()) |stride, i| {
            offset += pos[i] * stride;
        }
        var buf = self.getBuffer(T);
        buf[offset] = value;
    }

    pub fn setValue(self: Self, pos: []const u64, value: DTypeValue) void {
        switch (value) {
            .u8 => self.set(u8, pos, value.u8),
            .u64 => self.set(u64, pos, value.u64),
            .i64 => self.set(i64, pos, value.i64),
            .f32 => self.set(f32, pos, value.f32),
            .f64 => self.set(f64, pos, value.f64),
        }
    }

    pub fn retain(self: Self) void {
        if (self.ref_counter) |ref_counter| {
            ref_counter.increment();
        }
    }

    pub fn release(self: Self) void {
        if (self.ref_counter) |ref_counter| {
            // std.debug.print("release {}\n", .{ref_counter});
            if (ref_counter.decrement()) {
                // std.debug.print("dealloc: {*}\n", .{self.ref_counter});
                switch (self.buffer_union) {
                    .u8 => self.alc.?.free(self.buffer_union.u8),
                    .u64 => self.alc.?.free(self.buffer_union.u64),
                    .i64 => self.alc.?.free(self.buffer_union.i64),
                    .f32 => self.alc.?.free(self.buffer_union.f32),
                    .f64 => self.alc.?.free(self.buffer_union.f64),
                }
                self.alc.?.destroy(ref_counter);
            }
        }
    }

    fn formatElem(self: Self, writer: anytype, pos: []u64) !void {
        switch (self.buffer_union) {
            .u8 => try std.fmt.format(writer, "{}", .{self.get(u8, pos)}),
            .u64 => try std.fmt.format(writer, "{}", .{self.get(u64, pos)}),
            .i64 => try std.fmt.format(writer, "{}", .{self.get(i64, pos)}),
            .f32 => try std.fmt.format(writer, "{}", .{self.get(f32, pos)}),
            .f64 => try std.fmt.format(writer, "{}", .{self.get(f64, pos)}),
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
        try std.fmt.format(writer, "Array(dtype={}", .{dtypeToTypeName(self.buffer_union)});
        try writer.writeAll(", shape=(");
        for (self.getShape()) |s, i| {
            try std.fmt.format(writer, "{}", .{s});
            if (i < self.ndim - 1) {
                try writer.writeAll(", ");
            }
        }
        try writer.writeAll("), strides=(");
        for (self.getStrides()) |s, i| {
            try std.fmt.format(writer, "{}", .{s});
            if (i < self.ndim - 1) {
                try writer.writeAll(", ");
            }
        }
        try writer.writeAll("), data=");
        if (self.ndim == 0) {
            try self.formatElem(writer, &[_]u64{});
            return;
        }
        try writer.writeAll("\n");
        var pos = [_]u64{0} ** maxNumDim;
        const final_dim = self.ndim - 1;
        var dim: u64 = 0;
        var index: u64 = 0;
        try writer.writeAll("  [\n");
        var shape = self.getShape();
        while (true) {
            while (dim < final_dim) : (dim += 1) {
                try indent(writer, 2 * (dim + 2));
                try writer.writeAll("[");
                if (dim < final_dim - 1) {
                    try writer.writeAll("\n");
                }
            }
            var get_pos = pos[0..self.ndim];
            try self.formatElem(writer, get_pos);
            if (pos[final_dim] < shape[final_dim] - 1) {
                try writer.writeAll(", ");
            }
            index += 1;
            pos[final_dim] += 1;
            // carry
            while (dim > 0) {
                if (pos[dim] >= shape[dim]) {
                    pos[dim] = 0;
                    pos[dim - 1] += 1;
                    if (dim < final_dim) {
                        try indent(writer, 2 * (dim + 1));
                    }
                    try writer.writeAll("]");
                    try writer.writeAll(",\n");
                    dim -= 1;
                } else {
                    break;
                }
            }
            if (dim == 0 and pos[dim] == self.getShape()[dim]) {
                break;
            }
        }
        try writer.writeAll("  ]\n)");
    }
};

test "scalar" {
    const a = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{}, 1.0);
    defer a.release();
    std.testing.expect(a.numel == 1);
    std.testing.expect(a.ndim == 0);
    var it = a.createIterator();
    while (it.next()) |offset| {
        std.testing.expect(offset == 0);
    }

    const b = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{}, 2.0);
    defer b.release();
    const c = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{}, 0.0);
    defer c.release();
    plus(a, b, c);
    const d = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{}, 3.0);
    defer d.release();
    std.testing.expect(equal(c, d));
}

test "scalar_broadcast" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0, 1.0);
    defer a.release();
    const b = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{}, 2.0);
    defer b.release();
    const c = b.expandView(&[_]u64{ 2, 1, 3 });
    expectContiguous(c, false);
    const d = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 0.0);
    defer d.release();
    plus(a, c, d);
    var e_data = [_]f32{ 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const e = Array.fromBuffer(f32, &[_]u64{ 2, 1, 3 }, &e_data);
    std.testing.expect(equal(d, e));
    const f = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 0.0);
    defer f.release();
    plus(a, b, f);
    std.testing.expect(equal(f, e));
}

test "narrow" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0);
    defer a.release();
    expectContiguous(a, true);
    const b = a.narrowView(&[_]u64{ 0, 0, 0 }, &[_]u64{ 1, 1, 1 });
    expectContiguous(b, true);
    std.testing.expect(b.numel == 1);
    var buf = b.getBuffer(f32);
    std.testing.expect(buf[0] == 0.0);
    const c = a.narrowView(&[_]u64{ 0, 0, 0 }, &[_]u64{ 2, 3, 4 });
    expectContiguous(c, true);
    std.testing.expect(equal(a, c));
    const d = a.narrowView(&[_]u64{ 1, 2, 3 }, &[_]u64{ 1, 1, 1 });
    expectContiguous(d, true);
    const e = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 1, 1, 1 }, 23.0);
    defer e.release();
    std.testing.expect(equal(d, e));
    const f = a.narrowView(&[_]u64{ 1, 0, 0 }, &[_]u64{ 1, 3, 4 });
    const g = f.narrowView(&[_]u64{ 0, 2, 0 }, &[_]u64{ 1, 1, 4 });
    const h = g.narrowView(&[_]u64{ 0, 0, 3 }, &[_]u64{ 1, 1, 1 });
    std.testing.expect(equal(h, e));

    const i = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 0.0, 1.0);
    defer i.release();
    // select row
    const k = i.narrowView(&[_]u64{ 0, 0 }, &[_]u64{ 1, 3 });
    expectContiguous(k, true);
    // select column
    const j = i.narrowView(&[_]u64{ 0, 0 }, &[_]u64{ 2, 1 });
    expectContiguous(j, false);
}

test "reshape_view" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0);
    defer a.release();
    expectContiguous(a, true);
    const b = a.reshapeView(&[_]u64{ 1, 2, 1, 3, 1, 4, 1 });
    std.testing.expect(a.numel == b.numel);
    std.testing.expect(allcloseBuffers(f32, a, b, 0.0, 0.0));
}

test "zero size array" {
    const a = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{0}, 1.0);
    defer a.release();
    std.testing.expect(a.numel == 0);
    std.testing.expect(a.ndim == 1);
    var it = a.createIterator();
    while (it.next()) |offset| {
        std.testing.expect(false);
    }
}

test "alloc_with_string" {
    var output = try Array.allocWithString(f32, std.testing.allocator, "[[1, 2], [3, 4], [5, 6]]");
    defer output.release();
    var expected_output = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 3, 2 }, 1.0, 1.0);
    defer expected_output.release();
    std.testing.expect(equal(output, expected_output));
}

test "range" {
    var a_buf = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    const a = Array.fromBuffer(f32, &[_]u64{ 2, 1, 3 }, &a_buf);
    defer a.release();
    var b_buf = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    const b = Array.fromBuffer(f32, &[_]u64{ 1, 4, 1 }, &b_buf);
    defer b.release();

    const ar = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 0.0, 1.0);
    defer ar.release();
    std.testing.expect(equal(a, ar));
    const br = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 1, 4, 1 }, 0.0, 1.0);
    defer br.release();
    std.testing.expect(equal(b, br));
}

test "strides" {
    var strides = [_]u64{0} ** 3;
    Array.calculateStrides(&[_]u64{ 2, 1, 3 }, &strides);
    std.testing.expect(std.mem.eql(u64, &strides, &[_]u64{ 3, 3, 1 }));
    var t = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 3, 2, 1, 3 }, 0.0);
    defer t.release();
    var si = t.createIterator();
    var index: u64 = 0;
    while (si.next()) |v| {
        std.debug.assert(index == v);
        index += 1;
    }
    std.testing.expect(index == t.numel);
}

test "expand" {
    const t = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 1, 3, 1 }, 1.0);
    defer t.release();
    const t2 = t.expandView(&[_]u64{ 2, 8, 3, 9 });
    std.testing.expect(t.numel == 2 * 1 * 3 * 1);
    std.testing.expect(t2.numel == 2 * 8 * 3 * 9);
    var si = t2.createIterator();
    var numel: u64 = 0;
    while (si.next()) |_| {
        numel += 1;
    }
    std.testing.expect(numel == t2.numel);
}

test "contiguous" {
    const t = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0);
    defer t.release();
    std.testing.expect(t.is_contiguous);
    const et = t.expandView(&[_]u64{ 2, 4, 3 });
    std.testing.expect(!et.is_contiguous);
    std.testing.expect(std.mem.eql(f32, t.getBuffer(f32), t.getBuffer(f32)));
    const cet = try copyAlloc(std.testing.allocator, et);
    defer cet.release();
    std.testing.expect(cet.is_contiguous);
}

test "setvalue" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0);
    defer a.release();
    const b = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0);
    defer b.release();
    var pos = [_]u64{ 1, 1, 1 };
    b.setValue(&pos, a.getValue(&pos));
    std.testing.expect(b.get(f32, &pos) == 3 * 4 + 4 + 1);
}

test "format" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 0.0, 1.0);
    defer a.release();
    std.debug.print("a: {}\n", .{a});
}

pub fn assertDimsAreTheSame(x: Array, y: Array) void {
    if (x.ndim != y.ndim) {
        @panic("Arrays dimension counts do not match");
    }
}

pub fn assertTypesAreTheSame(x: Array, y: Array) void {
    if (x.dtype != y.dtype) {
        @panic("Arrays have different dtypes");
    }
}

pub fn assertShapesAreTheSame(x: anytype, y: anytype) void {
    if (!std.mem.eql(u64, x.getShape(), y.getShape())) {
        @panic("Arrays have differing shape");
    }
}

fn assertContiguous(x: anytype) void {
    if (!x.is_contiguous) {
        @panic("Array is not contiguous");
    }
}

fn checkContiguous(x: anytype) bool {
    if (x.numel == 0) {
        return true;
    }
    var it = x.createIterator();
    var prev_offset = it.next().?;
    while (it.next()) |offset| {
        if (prev_offset + 1 != offset) {
            return false;
        }
        prev_offset = offset;
    }
    return true;
}

pub fn zerosAlloc(alc: *std.mem.Allocator, dtype: DType, shape: []const u64) !Array {
    return switch (dtype) {
        .u8 => try Array.allocWithValue(u8, alc, shape, 0),
        .u64 => try Array.allocWithValue(u64, alc, shape, 0),
        .i64 => try Array.allocWithValue(i64, alc, shape, 0),
        .f32 => try Array.allocWithValue(f32, alc, shape, 0.0),
        .f64 => try Array.allocWithValue(f64, alc, shape, 0.0),
    };
}

pub fn zerosLikeAlloc(alc: *std.mem.Allocator, arr: Array) !Array {
    return zerosAlloc(alc, arr.dtype, arr.getShape());
}

pub fn onesAlloc(alc: *std.mem.Allocator, dtype: DType, shape: []const u64) !Array {
    return switch (dtype) {
        .u8 => try Array.allocWithValue(u8, alc, shape, 1),
        .u64 => try Array.allocWithValue(u64, alc, shape, 1),
        .i64 => try Array.allocWithValue(i64, alc, shape, 1),
        .f32 => try Array.allocWithValue(f32, alc, shape, 1.0),
        .f64 => try Array.allocWithValue(f64, alc, shape, 1.0),
    };
}

pub fn onesLikeAlloc(alc: *std.mem.Allocator, arr: Array) !Array {
    return onesAlloc(alc, arr.dtype, arr.getShape());
}

pub fn scalarAlloc(alc: *std.mem.Allocator, dtype: DType, value: f64) !Array {
    return switch (dtype) {
        .u8 => try Array.allocWithValue(u8, alc, &[_]u64{}, @floatToInt(u8, value)),
        .u64 => try Array.allocWithValue(u64, alc, &[_]u64{}, @floatToInt(u64, value)),
        .i64 => try Array.allocWithValue(i64, alc, &[_]u64{}, @floatToInt(i64, value)),
        .f32 => try Array.allocWithValue(f32, alc, &[_]u64{}, @floatCast(f32, value)),
        .f64 => try Array.allocWithValue(f64, alc, &[_]u64{}, value),
    };
}

fn fillUniformBuffer(comptime T: type, dst: Array, r: *std.rand.Random, low: Array, high: Array) void {
    var low_it = low.createIterator();
    var high_it = high.createIterator();
    var dst_it = dst.createIterator();
    var low_buf = low.getBuffer(T);
    var high_buf = high.getBuffer(T);
    var dst_buf = dst.getBuffer(T);
    while (dst_it.next()) |dst_offset| {
        var low_offset = low_it.next().?;
        var high_offset = high_it.next().?;
        var l = low_buf[low_offset];
        var h = high_buf[high_offset];
        if (l >= h) {
            @panic("Low is greater than or equal to high");
        }
        dst_buf[dst_offset] = switch (T) {
            u8, i64, u64 => r.intRangeAtMost(T, l, h),
            f32, f64 => r.float(T) * (h - l) + l,
            else => std.debug.panic("Invalid type {}", .{@typeName(T)}),
        };
    }
}

pub fn fillUniform(dst: Array, r: *std.rand.Random, low: Array, high: Array) void {
    assertTypesAreTheSame(dst, low);
    assertTypesAreTheSame(dst, high);
    const low_expanded = low.expandView(dst.getShape());
    const high_expanded = high.expandView(dst.getShape());
    switch (dst.dtype) {
        .u8 => fillUniformBuffer(u8, dst, r, low_expanded, high_expanded),
        .u64 => fillUniformBuffer(u64, dst, r, low_expanded, high_expanded),
        .i64 => fillUniformBuffer(i64, dst, r, low_expanded, high_expanded),
        .f32 => fillUniformBuffer(f32, dst, r, low_expanded, high_expanded),
        .f64 => fillUniformBuffer(f64, dst, r, low_expanded, high_expanded),
    }
}

fn copyBufferIterator(comptime T: type, src: Array, dst: Array) void {
    var src_it = src.createIterator();
    var dst_it = dst.createIterator();
    var src_buf = src.getBuffer(T);
    var dst_buf = dst.getBuffer(T);
    while (src_it.next()) |src_offset| {
        var dst_offset = dst_it.next().?;
        dst_buf[dst_offset] = src_buf[src_offset];
    }
}

pub fn copy(src: Array, dst: Array) void {
    assertTypesAreTheSame(src, dst);
    const src_expanded = src.expandView(dst.getShape());
    switch (src_expanded.buffer_union) {
        .u8 => copyBufferIterator(u8, src_expanded, dst),
        .u64 => copyBufferIterator(u64, src_expanded, dst),
        .i64 => copyBufferIterator(i64, src_expanded, dst),
        .f32 => copyBufferIterator(f32, src_expanded, dst),
        .f64 => copyBufferIterator(f64, src_expanded, dst),
    }
}

pub fn copyAlloc(alc: *std.mem.Allocator, src: Array) !Array {
    var dst = try zerosLikeAlloc(alc, src);
    copy(src, dst);
    return dst;
}

fn castBufferIterator(comptime SrcT: type, comptime DstT: type, src: Array, dst: Array) void {
    var src_it = src.createIterator();
    var dst_it = dst.createIterator();
    var src_buf = src.getBuffer(SrcT);
    var dst_buf = dst.getBuffer(DstT);
    while (src_it.next()) |src_offset| {
        var dst_offset = dst_it.next().?;
        var src_val = src_buf[src_offset];
        var dst_val = switch (@typeInfo(DstT)) {
            .Int => switch (@typeInfo(SrcT)) {
                .Int => @intCast(DstT, src_val),
                .Float => @floatToInt(DstT, src_val),
                else => @panic("unknown type"),
            },
            .Float => switch (@typeInfo(SrcT)) {
                .Int => @intToFloat(DstT, src_val),
                .Float => @floatCast(DstT, src_val),
                else => @panic("unknown type"),
            },
            else => @panic("unknown type"),
        };
        dst_buf[dst_offset] = dst_val;
    }
}

pub fn cast(src: Array, dst: Array) void {
    assertDimsAreTheSame(src, dst);
    assertShapesAreTheSame(src, dst);
    switch (src.buffer_union) {
        .u8 => switch (dst.buffer_union) {
            .u8 => castBufferIterator(u8, u8, src, dst),
            .u64 => castBufferIterator(u8, u64, src, dst),
            .i64 => castBufferIterator(u8, i64, src, dst),
            .f32 => castBufferIterator(u8, f32, src, dst),
            .f64 => castBufferIterator(u8, f64, src, dst),
        },
        .u64 => switch (dst.buffer_union) {
            .u8 => castBufferIterator(u64, u8, src, dst),
            .u64 => castBufferIterator(u64, u64, src, dst),
            .i64 => castBufferIterator(u64, i64, src, dst),
            .f32 => castBufferIterator(u64, f32, src, dst),
            .f64 => castBufferIterator(u64, f64, src, dst),
        },
        .i64 => switch (dst.buffer_union) {
            .u8 => castBufferIterator(i64, u8, src, dst),
            .u64 => castBufferIterator(i64, u64, src, dst),
            .i64 => castBufferIterator(i64, i64, src, dst),
            .f32 => castBufferIterator(i64, f32, src, dst),
            .f64 => castBufferIterator(i64, f64, src, dst),
        },
        .f32 => switch (dst.buffer_union) {
            .u8 => castBufferIterator(f32, u8, src, dst),
            .u64 => castBufferIterator(f32, u64, src, dst),
            .i64 => castBufferIterator(f32, i64, src, dst),
            .f32 => castBufferIterator(f32, f32, src, dst),
            .f64 => castBufferIterator(f32, f64, src, dst),
        },
        .f64 => switch (dst.buffer_union) {
            .u8 => castBufferIterator(f64, u8, src, dst),
            .u64 => castBufferIterator(f64, u64, src, dst),
            .i64 => castBufferIterator(f64, i64, src, dst),
            .f32 => castBufferIterator(f64, f32, src, dst),
            .f64 => castBufferIterator(f64, f64, src, dst),
        },
    }
}

pub fn castAlloc(alc: *std.mem.Allocator, src: Array, dtype: DType) !Array {
    if (src.dtype == dtype) {
        src.retain();
        return src;
    }
    var dst = try zerosAlloc(alc, dtype, src.getShape());
    cast(src, dst);
    return dst;
}

test "cast" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0, 1.0);
    defer a.release();
    const b = try Array.allocWithValue(f64, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 0.0);
    defer b.release();
    const c = try Array.allocWithRange(f64, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0, 1.0);
    defer c.release();
    cast(a, b);
    std.testing.expect(equal(b, c));
}

pub const DimArray = struct {
    ndim: u64,
    array: [maxNumDim]u64 = [_]u64{0} ** maxNumDim,
    const Self = @This();

    pub fn init(shape: []const u64) Self {
        var inst = Self{ .ndim = shape.len };
        std.mem.copy(u64, &inst.array, shape);
        return inst;
    }

    pub fn getSlice(self: *Self) []u64 {
        return self.array[0..self.ndim];
    }

    pub fn getConstSlice(self: *const Self) []const u64 {
        return self.array[0..self.ndim];
    }
};

pub fn binaryElementwiseShape(x_shape: []const u64, y_shape: []const u64) DimArray {
    var shape = DimArray{ .ndim = std.math.max(x_shape.len, y_shape.len) };
    // use the shape with a higher number of dimensions to populate our output shape
    var larger_shape = x_shape;
    var smaller_shape = y_shape;
    if (y_shape.len > x_shape.len) {
        larger_shape = y_shape;
        smaller_shape = x_shape;
    }
    std.mem.copy(u64, shape.array[0..], larger_shape);
    var offset: u64 = larger_shape.len - smaller_shape.len;
    for (smaller_shape) |s| {
        if (!(shape.array[offset] == s or shape.array[offset] == 1 or s == 1)) {
            @panic("Shapes for each dimension must be equal or one of them must be 1");
        }
        shape.array[offset] = std.math.max(shape.array[offset], s);
        offset += 1;
    }
    return shape;
}

pub const BinaryElementwiseOperation = enum {
    plus,
    minus,
    times,
    divide,
    power,
    max,
    gt,
    gte,
    lt,
    lte,
    eq,
};

fn boolToValue(comptime T: type, b: bool) T {
    if (b) {
        return 1;
    } else {
        return 0;
    }
}

fn binaryElementwiseOperationOnBuffers(comptime T: type, x_in: Array, y_in: Array, z_out: Array, op: BinaryElementwiseOperation) void {
    var si_x = x_in.createIterator();
    var si_y = y_in.createIterator();
    var si_z = z_out.createIterator();
    var x_buf = x_in.getBuffer(T);
    var y_buf = y_in.getBuffer(T);
    var z_buf = z_out.getBuffer(T);
    while (si_x.next()) |x_offset| {
        var y_offset = si_y.next().?;
        var z_offset = si_z.next().?;
        z_buf[z_offset] = switch (op) {
            .plus => x_buf[x_offset] + y_buf[y_offset],
            .minus => x_buf[x_offset] - y_buf[y_offset],
            .times => x_buf[x_offset] * y_buf[y_offset],
            .divide => switch (@typeInfo(T)) {
                .Int => @divTrunc(x_buf[x_offset], y_buf[y_offset]),
                else => x_buf[x_offset] / y_buf[y_offset],
            },
            .power => std.math.pow(T, x_buf[x_offset], y_buf[y_offset]),
            .max => std.math.max(x_buf[x_offset], y_buf[y_offset]),
            .eq => boolToValue(T, x_buf[x_offset] == y_buf[y_offset]),
            .gt => boolToValue(T, x_buf[x_offset] > y_buf[y_offset]),
            .gte => boolToValue(T, x_buf[x_offset] >= y_buf[y_offset]),
            .lt => boolToValue(T, x_buf[x_offset] < y_buf[y_offset]),
            .lte => boolToValue(T, x_buf[x_offset] <= y_buf[y_offset]),
        };
    }
}

fn binaryElementwiseOperation(x_in: Array, y_in: Array, z_out: Array, op: BinaryElementwiseOperation) void {
    assertTypesAreTheSame(x_in, y_in);
    assertTypesAreTheSame(x_in, z_out);
    var out_shape = binaryElementwiseShape(x_in.getShape(), y_in.getShape());
    const x_in_expanded = x_in.expandView(out_shape.getSlice());
    const y_in_expanded = y_in.expandView(out_shape.getSlice());
    assertShapesAreTheSame(x_in_expanded, y_in_expanded);
    if (!std.mem.eql(u64, z_out.getShape(), out_shape.getSlice())) {
        @panic("Attempted to use output Array with wrong shape");
    }
    switch (x_in.dtype) {
        .u8 => binaryElementwiseOperationOnBuffers(u8, x_in_expanded, y_in_expanded, z_out, op),
        .u64 => binaryElementwiseOperationOnBuffers(u64, x_in_expanded, y_in_expanded, z_out, op),
        .i64 => binaryElementwiseOperationOnBuffers(i64, x_in_expanded, y_in_expanded, z_out, op),
        .f32 => binaryElementwiseOperationOnBuffers(f32, x_in_expanded, y_in_expanded, z_out, op),
        .f64 => binaryElementwiseOperationOnBuffers(f64, x_in_expanded, y_in_expanded, z_out, op),
    }
}

/// Add 2 Arrays together, putting the result in the 3rd
pub fn plus(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.plus);
}

test "plus" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0, 1.0);
    defer a.release();
    const b = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 1, 4, 1 }, 1.0, 1.0);
    defer b.release();
    const c = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 4, 3 }, 0.0);
    defer c.release();
    plus(a, b, c);
    std.testing.expect(c.numel == 2 * 4 * 3);
    var d_data = [_]f32{ 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0 };
    const d = Array.fromBuffer(f32, &[_]u64{ 2, 4, 3 }, &d_data);
    std.testing.expect(equal(c, d));
    const e = try plusAlloc(std.testing.allocator, a, b);
    defer e.release();
    std.testing.expect(equal(c, e));
}

pub fn minus(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.minus);
}

pub fn times(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.times);
}

pub fn divide(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.divide);
}

pub fn power(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.power);
}

test "power" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0, 1.0);
    defer a.release();
    const b = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 1, 4, 1 }, 1.0, 1.0);
    defer b.release();
    const c = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 4, 3 }, 0.0);
    defer c.release();
    power(a, b, c);
    std.testing.expect(c.numel == 2 * 4 * 3);
    var d_data = [_]f32{
        1.000e+00, 2.000e+00, 3.000e+00, 1.000e+00, 4.000e+00, 9.000e+00,
        1.000e+00, 8.000e+00, 2.700e+01, 1.000e+00, 1.600e+01, 8.100e+01,
        4.000e+00, 5.000e+00, 6.000e+00, 1.600e+01, 2.500e+01, 3.600e+01,
        6.400e+01, 1.250e+02, 2.160e+02, 2.560e+02, 6.250e+02, 1.296e+03,
    };
    const d = Array.fromBuffer(f32, &[_]u64{ 2, 4, 3 }, &d_data);
    std.testing.expect(equal(c, d));
    const e = try powerAlloc(std.testing.allocator, a, b);
    defer e.release();
    std.testing.expect(equal(c, e));
}

pub fn max(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.max);
}

test "max" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 0.0, 1.0);
    defer a.release();
    const b = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 5.0, -1.0);
    defer b.release();
    const c = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 0.0);
    defer c.release();
    max(a, b, c);
    var d_data = [_]f32{
        5.0, 4.0, 3.0, 3.0, 4.0, 5.0,
    };
    const d = Array.fromBuffer(f32, &[_]u64{ 2, 3 }, &d_data);
    std.testing.expect(equal(c, d));
    const e = try maxAlloc(std.testing.allocator, a, b);
    defer e.release();
    std.testing.expect(equal(e, d));
}

pub fn gt(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.gt);
}

pub fn gte(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.gte);
}

pub fn eq(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.eq);
}

pub fn lt(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.lt);
}

pub fn lte(x_in: Array, y_in: Array, z_out: Array) void {
    binaryElementwiseOperation(x_in, y_in, z_out, BinaryElementwiseOperation.lte);
}

test "comparison" {
    const ComparisonFn = fn (a: Array, b: Array, c: Array) void;
    const ComparisonFnAlloc = fn (alc: *std.mem.Allocator, a: Array, b: Array) error{OutOfMemory}!Array;
    const TestCase = struct {
        func: ComparisonFn,
        func_alloc: ComparisonFnAlloc,
        expected_output: []const u8,
    };

    var testcases = [_]TestCase{
        TestCase{
            .func = gt,
            .func_alloc = gtAlloc,
            .expected_output = "[0.0, 0.0, 0.0, 1.0, 1.0]",
        },
        TestCase{
            .func = gte,
            .func_alloc = gteAlloc,
            .expected_output = "[0.0, 0.0, 1.0, 1.0, 1.0]",
        },
        TestCase{
            .func = eq,
            .func_alloc = eqAlloc,
            .expected_output = "[0.0, 0.0, 1.0, 0.0, 0.0]",
        },
        TestCase{
            .func = lt,
            .func_alloc = ltAlloc,
            .expected_output = "[1.0, 1.0, 0.0, 0.0, 0.0]",
        },
        TestCase{
            .func = lte,
            .func_alloc = lteAlloc,
            .expected_output = "[1.0, 1.0, 1.0, 0.0, 0.0]",
        },
    };

    for (testcases) |tc| {
        const first_input = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{5}, 0.0, 1.0);
        defer first_input.release();
        const second_input = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{5}, 4.0, -1.0);
        defer second_input.release();
        const output = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{5}, 0.0);
        defer output.release();
        tc.func(first_input, second_input, output);
        var expected_output = try Array.allocWithString(f32, std.testing.allocator, tc.expected_output);
        defer expected_output.release();
        std.testing.expect(equal(output, expected_output));
        const output2 = try tc.func_alloc(std.testing.allocator, first_input, second_input);
        defer output2.release();
        std.testing.expect(equal(output2, expected_output));
    }
}

/// Add 2 Arrays together, allocating a result Array for the output
pub fn binaryElementwiseOperationAlloc(alc: *std.mem.Allocator, x: Array, y: Array, op: BinaryElementwiseOperation) !Array {
    var z_shape = binaryElementwiseShape(x.getShape(), y.getShape());
    var z = try zerosAlloc(alc, x.dtype, z_shape.getSlice());
    binaryElementwiseOperation(x, y, z, op);
    return z;
}

pub fn plusAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.plus);
}

pub fn minusAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.minus);
}

pub fn timesAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.times);
}

pub fn divideAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.divide);
}

pub fn powerAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.power);
}

pub fn maxAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.max);
}

pub fn eqAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.eq);
}

pub fn gtAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.gt);
}

pub fn gteAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.gte);
}

pub fn ltAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.lt);
}

pub fn lteAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    return try binaryElementwiseOperationAlloc(alc, x, y, BinaryElementwiseOperation.lte);
}

pub const UnaryElementwiseOperation = enum {
    uminus,
    log,
    log2,
    exp,
};

fn unaryElementwiseOperationOnBuffers(comptime T: type, x_in: Array, z_out: Array, op: UnaryElementwiseOperation) void {
    var si_x = x_in.createIterator();
    var si_z = z_out.createIterator();
    var x_buf = x_in.getBuffer(T);
    var z_buf = z_out.getBuffer(T);
    while (si_x.next()) |x_offset| {
        var z_offset = si_z.next().?;
        z_buf[z_offset] = switch (@typeInfo(T)) {
            .Int => |int| switch (op) {
                .uminus => switch (int.is_signed) {
                    true => -x_buf[x_offset],
                    false => @panic("uminus on unsigned int"),
                },
                .log => @floatToInt(T, std.math.log(f64, std.math.e, @intToFloat(f64, x_buf[x_offset]))),
                .log2 => @floatToInt(T, std.math.log(f64, 2.0, @intToFloat(f64, x_buf[x_offset]))),
                .exp => @floatToInt(T, std.math.pow(f64, std.math.e, @intToFloat(f64, x_buf[x_offset]))),
            },
            .Float => switch (op) {
                .uminus => -x_buf[x_offset],
                .log => std.math.log(T, std.math.e, x_buf[x_offset]),
                .log2 => std.math.log(T, 2.0, x_buf[x_offset]),
                .exp => std.math.pow(T, std.math.e, x_buf[x_offset]),
            },
            else => @panic("unrecognized type"),
        };
    }
}

fn unaryElementwiseOperation(x_in: Array, z_out: Array, op: UnaryElementwiseOperation) void {
    assertTypesAreTheSame(x_in, z_out);
    if (!std.mem.eql(u64, z_out.getShape(), x_in.getShape())) {
        @panic("Attempted to use output Array with wrong shape");
    }
    switch (x_in.dtype) {
        .u8 => unaryElementwiseOperationOnBuffers(u8, x_in, z_out, op),
        .u64 => unaryElementwiseOperationOnBuffers(u64, x_in, z_out, op),
        .i64 => unaryElementwiseOperationOnBuffers(i64, x_in, z_out, op),
        .f32 => unaryElementwiseOperationOnBuffers(f32, x_in, z_out, op),
        .f64 => unaryElementwiseOperationOnBuffers(f64, x_in, z_out, op),
    }
}
pub fn unaryElementwiseOperationAlloc(alc: *std.mem.Allocator, x: Array, op: UnaryElementwiseOperation) !Array {
    var z = try zerosAlloc(alc, x.dtype, x.getShape());
    unaryElementwiseOperation(x, z, op);
    return z;
}

pub fn uplus(x_in: Array, z_out: Array) void {
    copy(x_in, z_out);
}

pub fn uplusAlloc(alc: *std.mem.Allocator, x: Array) !Array {
    return try copyAlloc(alc, x);
}

test "uplus" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0, 1.0);
    defer a.release();
    const b = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 0.0);
    defer b.release();
    uplus(a, b);
    std.testing.expect(equal(a, b));
}

pub fn uminus(x_in: Array, z_out: Array) void {
    unaryElementwiseOperation(x_in, z_out, UnaryElementwiseOperation.uminus);
}

pub fn uminusAlloc(alc: *std.mem.Allocator, x: Array) !Array {
    return try unaryElementwiseOperationAlloc(alc, x, UnaryElementwiseOperation.uminus);
}

pub fn log(x_in: Array, z_out: Array) void {
    unaryElementwiseOperation(x_in, z_out, UnaryElementwiseOperation.ln);
}

pub fn logAlloc(alc: *std.mem.Allocator, x: Array) !Array {
    return try unaryElementwiseOperationAlloc(alc, x, UnaryElementwiseOperation.log);
}

pub fn log2(x_in: Array, z_out: Array) void {
    unaryElementwiseOperation(x_in, z_out, UnaryElementwiseOperation.log2);
}

pub fn log2Alloc(alc: *std.mem.Allocator, x: Array) !Array {
    return try unaryElementwiseOperationAlloc(alc, x, UnaryElementwiseOperation.log2);
}

pub fn exp(x_in: Array, z_out: Array) void {
    unaryElementwiseOperation(x_in, z_out, UnaryElementwiseOperation.exp);
}

pub fn expAlloc(alc: *std.mem.Allocator, x: Array) !Array {
    return try unaryElementwiseOperationAlloc(alc, x, UnaryElementwiseOperation.exp);
}

fn transposeBuffer(comptime T: type, src: Array, dst: Array) void {
    var src_strides = src.getStrides();
    var src_strides_reversed_shape = DimArray.init(src_strides);
    reverseSlice(u64, src_strides, src_strides_reversed_shape.getSlice());
    var src_it = StridedIterator.init(dst.getShape(), src_strides_reversed_shape.getSlice(), src.offset);
    var dst_it = dst.createIterator();
    var src_buf = src.getBuffer(T);
    var dst_buf = dst.getBuffer(T);
    if (&src_buf[0] == &dst_buf[0]) {
        @panic("src and dst buf point to the same data, but transpose cannot be run in place");
    }
    while (src_it.next()) |src_offset| {
        var dst_offset = dst_it.next().?;
        dst_buf[dst_offset] = src_buf[src_offset];
    }
}

pub fn reverseSlice(comptime T: type, in: []const T, out: []T) void {
    for (in) |s, i| {
        out[out.len - 1 - i] = s;
    }
}

pub fn transpose(x_in: Array, z_out: Array) void {
    assertTypesAreTheSame(x_in, z_out);
    var reversed_shape = DimArray.init(x_in.getShape());
    reverseSlice(u64, x_in.getShape(), reversed_shape.getSlice());
    if (!std.mem.eql(u64, z_out.getShape(), reversed_shape.getSlice())) {
        @panic("output array for transpose has incorrect shape");
    }
    switch (x_in.dtype) {
        .u8 => transposeBuffer(u8, x_in, z_out),
        .u64 => transposeBuffer(u64, x_in, z_out),
        .i64 => transposeBuffer(i64, x_in, z_out),
        .f32 => transposeBuffer(f32, x_in, z_out),
        .f64 => transposeBuffer(f64, x_in, z_out),
    }
}

pub fn transposeAlloc(alc: *std.mem.Allocator, x: Array) !Array {
    var reversed_shape = DimArray.init(x.getShape());
    reverseSlice(u64, x.getShape(), reversed_shape.getSlice());
    var z = try zerosAlloc(alc, x.dtype, reversed_shape.getSlice());
    transpose(x, z);
    return z;
}

test "transpose" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0);
    defer a.release();
    const b = try zerosAlloc(std.testing.allocator, a.dtype, &[_]u64{ 4, 3, 2 });
    defer b.release();
    transpose(a, b);
    var c_data = [_]f32{
        0,  12, 4,  16, 8,  20, 1,  13, 5, 17, 9, 21, 2, 14, 6, 18, 10,
        22, 3,  15, 7,  19, 11, 23,
    };
    const c = Array.fromBuffer(f32, &[_]u64{ 4, 3, 2 }, &c_data);
    std.testing.expect(equal(b, c));
}

fn copyBufferDirect(comptime T: type, src: Array, dst: Array) void {
    var src_buf = src.getBuffer(T);
    var dst_buf = dst.getBuffer(T);
    std.mem.copy(T, dst_buf, src_buf);
}

/// Reshape an Array, the new shape must have the same number of elements
pub fn reshape(in: Array, out: Array) void {
    assertTypesAreTheSame(in, out);
    if (in.numel != out.numel) {
        @panic("Input and output Arrays have differing number of elements");
    }
    switch (in.buffer_union) {
        .u8 => copyBufferDirect(u8, in, out),
        .u64 => copyBufferDirect(u64, in, out),
        .i64 => copyBufferDirect(i64, in, out),
        .f32 => copyBufferDirect(f32, in, out),
        .f64 => copyBufferDirect(f64, in, out),
    }
}

/// Reshape an Array to the provided shape, allocating an Array to hold the result
pub fn reshapeAlloc(alc: *std.mem.Allocator, in: Array, shape: []const u64) !Array {
    var out = try zerosAlloc(alc, in.dtype, shape);
    reshape(in, out);
    return out;
}

test "reshape" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 0.0, 1.0);
    defer a.release();
    const b = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 1, 3, 2, 1 }, 0.0);
    defer b.release();
    reshape(a, b);
    std.testing.expect(std.mem.eql(f32, a.getBuffer(f32), b.getBuffer(f32)));
    var shape = [_]u64{ 1, 3, 2, 1 };
    const c = try reshapeAlloc(std.testing.allocator, a, &shape);
    defer c.release();
    std.testing.expect(std.mem.eql(f32, a.getBuffer(f32), c.getBuffer(f32)));

    var a_it = a.createIterator();
    var b_it = b.createIterator();
    var a_buf = a.getBuffer(f32);
    var b_buf = b.getBuffer(f32);
    while (a_it.next()) |a_offset| {
        var b_offset = b_it.next().?;
        std.testing.expect(a_buf[a_offset] == b_buf[b_offset]);
    }
}

fn mtimesShape(x_shape: []const u64, y_shape: []const u64) DimArray {
    if (x_shape.len != 2 or y_shape.len != 2) {
        @panic("Matmul arguments must have 2 dimensions each");
    }
    if (x_shape[1] != y_shape[0]) {
        @panic("Dimension mismatch for matrix times");
    }
    return DimArray.init(&[2]u64{ x_shape[0], y_shape[1] });
}

pub fn mtimes(x_in: Array, y_in: Array, z_out: Array) void {
    assertContiguous(z_out);
    assertTypesAreTheSame(x_in, y_in);
    assertTypesAreTheSame(x_in, z_out);

    if (!(x_in.getShape()[1] == y_in.getShape()[0])) {
        @panic("Shapes for reduced dimension must be equal");
    }
    var out_shape = mtimesShape(x_in.getShape(), y_in.getShape());

    if (!std.mem.eql(u64, z_out.getShape(), out_shape.getSlice())) {
        @panic("Output shape incorrect");
    }
    switch (x_in.buffer_union) {
        .u8 => mtimesBuffers(u8, x_in, y_in, z_out),
        .u64 => mtimesBuffers(u64, x_in, y_in, z_out),
        .i64 => mtimesBuffers(i64, x_in, y_in, z_out),
        .f32 => mtimesBuffers(f32, x_in, y_in, z_out),
        .f64 => mtimesBuffers(f64, x_in, y_in, z_out),
    }
}

fn mklMtimesBuffers(comptime T: type, x_in: Array, y_in: Array, z_out: Array) void {
    var m = x_in.getShape()[0];
    var k = x_in.getShape()[1];
    var n = y_in.getShape()[1];

    var x_buf = x_in.getBuffer(T);
    var y_buf = y_in.getBuffer(T);
    var z_buf = z_out.getBuffer(T);
    var lda: u64 = k;
    var ldb: u64 = n;
    var ldc: u64 = n;
    var alpha: T = 1.0;
    var beta: T = 1.0;
    switch (T) {
        f32 => mkl.cblas_sgemm(&x_buf[x_in.offset], &y_buf[y_in.offset], &z_buf[z_out.offset], lda, ldb, ldc, m, n, k, alpha, beta),
        f64 => mkl.cblas_dgemm(&x_buf[x_in.offset], &y_buf[y_in.offset], &z_buf[z_out.offset], lda, ldb, ldc, m, n, k, alpha, beta),
        else => @compileError("Unsupported type for MKL mtimes"),
    }
}

fn mklMtimes(x_in: Array, y_in: Array, z_out: Array) void {
    assertTypesAreTheSame(x_in, y_in);
    assertTypesAreTheSame(x_in, z_out);
    assertContiguous(x_in);
    assertContiguous(y_in);
    assertContiguous(z_out);

    if (!(x_in.getShape()[1] == y_in.getShape()[0])) {
        @panic("Shapes for reduced dimension must be equal");
    }
    var out_shape = mtimesShape(x_in.getShape(), y_in.getShape());

    if (!std.mem.eql(u64, z_out.getShape(), out_shape.getSlice())) {
        @panic("Output shape incorrect");
    }

    switch (x_in.dtype) {
        .f32 => mklMtimesBuffers(f32, x_in, y_in, z_out),
        .f64 => mklMtimesBuffers(f64, x_in, y_in, z_out),
        else => std.debug.panic("Unsupported dtype for MKL mtimes {}", .{x_in.dtype}),
    }
}

test "cblas_sgemm" {
    if (USE_MKL) {
        {
            var a: f32 = 2.0;
            var b: f32 = 2.0;
            var c: f32 = 2.0;
            mkl.cblas_sgemm(&a, &b, &c, 4, 4, 4, 1, 1, 1, 1.0, 1.0);
            std.testing.expect(c == 6.0);
        }
        {
            var a = try Array.allocWithString(f32, std.testing.allocator, "[[1, 2], [3, 4], [5, 6]]");
            defer a.release();
            var b = try Array.allocWithString(f32, std.testing.allocator, "[[1, 2, 3, 4], [5, 6, 7, 8]]");
            defer b.release();
            var c = try Array.allocWithString(f32, std.testing.allocator, "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]");
            defer c.release();
            mklMtimes(a, b, c);
            var expected_output = try Array.allocWithString(f32, std.testing.allocator, "[[12, 16, 20, 24], [28, 36, 44, 52], [44, 56, 68, 80]]");
            defer expected_output.release();
            std.testing.expect(equal(c, expected_output));
        }
    }
}

fn mtimesBuffers(comptime T: type, x_in: Array, y_in: Array, z_out: Array) void {
    var z_r: u64 = 0;
    while (z_r < z_out.getShape()[0]) : (z_r += 1) {
        var z_c: u64 = 0;
        while (z_c < z_out.getShape()[1]) : (z_c += 1) {
            var total: T = z_out.get(T, &[2]u64{ z_r, z_c });
            var i: u64 = 0;
            while (i < x_in.getShape()[1]) : (i += 1) {
                total += x_in.get(T, &[2]u64{ z_r, i }) * y_in.get(T, &[2]u64{ i, z_c });
            }
            z_out.set(T, &[2]u64{ z_r, z_c }, total);
        }
    }
}

pub fn mtimesAlloc(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    var z_shape = mtimesShape(x.getShape(), y.getShape());
    var z = try zerosAlloc(alc, x.dtype, z_shape.getSlice());
    if (USE_MKL) {
        mklMtimes(x, y, z);
    } else {
        mtimes(x, y, z);
    }
    return z;
}

test "mtimes" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 0.0, 1.0);
    defer a.release();
    const b = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 3, 2 }, 0.0, 1.0);
    defer b.release();
    const c = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 2 }, 0.0);
    defer c.release();
    mtimes(a, b, c);
    var d_data = [_]f32{ 10, 13, 28, 40 };
    const d = Array.fromBuffer(f32, &[_]u64{ 2, 2 }, &d_data);
    std.testing.expect(equal(c, d));
    const e = try mtimesAlloc(std.testing.allocator, a, b);
    defer e.release();
    std.testing.expect(equal(c, e));
}

fn reduceShape(x_shape: []const u64, dims: []const u64, keepdims: bool) DimArray {
    var ndim: u64 = 0;
    if (keepdims) {
        ndim = x_shape.len;
    } else {
        ndim = x_shape.len - dims.len;
    }
    for (dims) |d| {
        if (d >= x_shape.len) {
            @panic("invalid dimension for reduce");
        }
    }
    var shape = DimArray{ .ndim = ndim };
    var i: u64 = 0;
    for (x_shape) |s, d| {
        if (keepdims) {
            if (contains(u64, dims, d)) {
                shape.array[i] = 1;
            } else {
                shape.array[i] = s;
            }
            i += 1;
        } else {
            if (!contains(u64, dims, d)) {
                shape.array[i] = s;
                i += 1;
            }
        }
    }
    return shape;
}

pub const ReduceOperation = enum {
    sum,
    max,
    mean,
};

fn reduceBuffers(comptime T: type, in: Array, in_it: *StridedIterator, out: Array, out_offset: u64, op: ReduceOperation) void {
    var r: T = 0;
    var in_buf = in.getBuffer(T);
    var out_buf = out.getBuffer(T);
    while (in_it.next()) |offset| {
        r = switch (op) {
            .sum => r + in_buf[offset],
            .max => std.math.max(r, in_buf[offset]),
            .mean => r + in_buf[offset],
        };
    }
    if (op == .mean) {
        r = switch (@typeInfo(T)) {
            .Int => @divTrunc(r, @intCast(T, in.numel)),
            .Float => r / @intToFloat(T, in.numel),
            else => @panic("invalid type"),
        };
    }
    out_buf[out_offset] = r;
}

/// Find the sum of an Array, removes reduced dimensions
pub fn reduceSum(in: Array, out: Array, dims: []const u64) void {
    reduce(in, out, dims, false, .sum);
}

/// Find the sum of an Array, keep reduced dimensions
pub fn keepSum(in: Array, out: Array, dims: []const u64) void {
    reduce(in, out, dims, true, .sum);
}

pub fn reduce(in: Array, out: Array, dims: []const u64, keepdims: bool, op: ReduceOperation) void {
    assertTypesAreTheSame(in, out);
    if (in.ndim < dims.len) {
        @panic("Invalid number of dims to sum across");
    }

    var out_ndim = in.ndim;
    if (!keepdims) {
        out_ndim = in.ndim - dims.len;
    }
    if (out_ndim != out.ndim) {
        @panic("Output has wrong number of dimensions");
    }

    var out_shape = reduceShape(in.getShape(), dims, keepdims);
    if (!std.mem.eql(u64, out.getShape(), out_shape.getSlice())) {
        @panic("Output has wrong shape");
    }

    var iterShape = DimArray.init(in.getShape());
    var reducedShape = DimArray.init(in.getShape());
    var d: u64 = 0;
    while (d < in.ndim) : (d += 1) {
        if (contains(u64, dims, d)) {
            iterShape.array[d] = 1;
        } else {
            reducedShape.array[d] = 1;
        }
    }
    var in_it = PositionIterator.init(iterShape.getSlice());
    var out_it = out.createIterator();
    while (in_it.next()) |pos| {
        const inner = in.narrowView(pos, reducedShape.getSlice());
        var inner_it = inner.createIterator();
        var out_offset = out_it.next().?;
        switch (in.buffer_union) {
            .u8 => reduceBuffers(u8, in, &inner_it, out, out_offset, op),
            .u64 => reduceBuffers(u64, in, &inner_it, out, out_offset, op),
            .i64 => reduceBuffers(i64, in, &inner_it, out, out_offset, op),
            .f32 => reduceBuffers(f32, in, &inner_it, out, out_offset, op),
            .f64 => reduceBuffers(f64, in, &inner_it, out, out_offset, op),
        }
    }
}

/// Find the max of an Array, removes reduced dimensions
pub fn reduceMax(in: Array, out: Array, dims: []const u64) void {
    reduce(in, out, dims, false, .max);
}

/// Find the max of an Array, keep reduced dimensions
pub fn keepMax(in: Array, out: Array, dims: []const u64) void {
    reduce(in, out, dims, true, .max);
}

pub fn reduceMean(in: Array, out: Array, dims: []const u64) void {
    reduce(in, out, dims, false, .mean);
}

pub fn keepMean(in: Array, out: Array, dims: []const u64) void {
    reduce(in, out, dims, true, .mean);
}

fn findBroadcastDims(input_shape: []const u64, output_shape: []const u64) DimArray {
    if (output_shape.len < input_shape.len) {
        @panic("invalid shapes");
    }
    // copy the initial dims from the output shape (that are missing on the input shape)
    var offset = output_shape.len - input_shape.len;
    var result = DimArray{ .ndim = 0 };
    while (result.ndim < offset) {
        result.array[result.ndim] = result.ndim;
        result.ndim += 1;
    }
    // add in any existing dims that are broadcast
    for (input_shape) |_, d| {
        if (input_shape[d] == 1 and output_shape[d + offset] != 1) {
            result.array[result.ndim] = d + offset;
            result.ndim += 1;
        }
    }
    return result;
}

test "find_bcast_dims" {
    {
        var dims_shape = findBroadcastDims(&[_]u64{3}, &[_]u64{ 3, 3 });
        std.testing.expect(std.mem.eql(u64, dims_shape.getSlice(), &[_]u64{0}));
    }
    {
        var dims_shape = findBroadcastDims(&[_]u64{ 1, 3 }, &[_]u64{ 3, 3 });
        std.testing.expect(std.mem.eql(u64, dims_shape.getSlice(), &[_]u64{0}));
    }
    {
        var dims_shape = findBroadcastDims(&[_]u64{1}, &[_]u64{ 3, 3 });
        std.testing.expect(std.mem.eql(u64, dims_shape.getSlice(), &[_]u64{ 0, 1 }));
    }
    {
        var dims_shape = findBroadcastDims(&[_]u64{ 1, 1 }, &[_]u64{ 3, 3 });
        std.testing.expect(std.mem.eql(u64, dims_shape.getSlice(), &[_]u64{ 0, 1 }));
    }
    {
        var dims_shape = findBroadcastDims(&[_]u64{}, &[_]u64{ 3, 3 });
        std.testing.expect(std.mem.eql(u64, dims_shape.getSlice(), &[_]u64{ 0, 1 }));
    }
}

/// Do a sum where we find the reduced dimensions by following broadcasting rules
/// this assumes that `out` was broadcast to the shape of `in`, so do the reverse
/// mapping
pub fn bcastsum(in: Array, out: Array) void {
    // "in" here is the post broadcast shape
    // "out" is the pre-broadcast shape
    //
    // in.shape = (3,3)
    // out.shape = (1,3) or (3)
    // dims_shape = (0)
    //
    // in.shape = (3,3)
    // out.shape = (1,1)
    // dims_shape = (0, 1)
    var dims_shape = findBroadcastDims(out.getShape(), in.getShape());
    var expanded_shape = DimArray.init(in.getShape());
    for (dims_shape.getSlice()) |d| {
        expanded_shape.array[d] = 1;
    }
    var expanded_out = out.reshapeView(expanded_shape.getSlice());
    reduce(in, expanded_out, dims_shape.getSlice(), true, ReduceOperation.sum);
}

pub fn reduceAlloc(alc: *std.mem.Allocator, in: Array, dims: []const u64, keepdims: bool, op: ReduceOperation) !Array {
    var out_shape = reduceShape(in.getShape(), dims, keepdims);
    var out = try zerosAlloc(alc, in.dtype, out_shape.getSlice());
    reduce(in, out, dims, keepdims, op);
    return out;
}

pub fn reduceSumAlloc(alc: *std.mem.Allocator, in: Array, dims: []const u64) !Array {
    return reduceAlloc(alc, in, dims, false, .sum);
}

pub fn keepSumAlloc(alc: *std.mem.Allocator, in: Array, dims: []const u64) !Array {
    return reduceAlloc(alc, in, dims, true, .sum);
}

test "reduce_sum" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0);
    defer a.release();
    const b = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3 }, 0.0);
    defer b.release();
    reduceSum(a, b, &[_]u64{2});
    var c_data = [_]f32{ 6.0, 22.0, 38.0, 54.0, 70.0, 86.0 };
    const c = Array.fromBuffer(f32, &[_]u64{ 2, 3 }, &c_data);
    std.testing.expect(equal(b, c));
    const d = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 4 }, 0.0);
    defer d.release();
    reduceSum(a, d, &[_]u64{1});
    var e_data = [_]f32{ 12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0 };
    const e = Array.fromBuffer(f32, &[_]u64{ 2, 4 }, &e_data);
    std.testing.expect(equal(d, e));

    const f = try reduceSumAlloc(std.testing.allocator, a, &[_]u64{2});
    defer f.release();
    std.testing.expect(equal(c, f));
}

test "keep_sum" {
    const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0);
    defer a.release();
    const b = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 1 }, 0.0);
    defer b.release();
    keepSum(a, b, &[_]u64{2});
    var c_data = [_]f32{ 6.0, 22.0, 38.0, 54.0, 70.0, 86.0 };
    const c = Array.fromBuffer(f32, &[_]u64{ 2, 3, 1 }, &c_data);
    std.testing.expect(equal(b, c));
}

pub fn reduceMaxAlloc(alc: *std.mem.Allocator, in: Array, dims: []const u64) !Array {
    return reduceAlloc(alc, in, dims, false, .max);
}

pub fn keepMaxAlloc(alc: *std.mem.Allocator, in: Array, dims: []const u64) !Array {
    return reduceAlloc(alc, in, dims, true, .max);
}

pub fn reduceMeanAlloc(alc: *std.mem.Allocator, in: Array, dims: []const u64) !Array {
    return reduceAlloc(alc, in, dims, false, .mean);
}

pub fn keepMeanAlloc(alc: *std.mem.Allocator, in: Array, dims: []const u64) !Array {
    return reduceAlloc(alc, in, dims, true, .mean);
}

fn argMaxBuffer(comptime T: type, in: Array, in_it: *PositionIterator, dim: u64) u64 {
    var m: T = 0;
    var idx: u64 = 0;
    while (in_it.next()) |pos| {
        var v = in.get(T, pos);
        if (v > m or pos[dim] == 0) {
            m = v;
            idx = pos[dim];
        }
    }
    return idx;
}

fn argMax(in: Array, out: Array, dim: u64, keepdims: bool) void {
    if (out.dtype != .u64) {
        @panic("Output dtype must be u64");
    }

    var out_ndim = in.ndim;
    if (!keepdims) {
        out_ndim = in.ndim - 1;
    }
    if (out_ndim != out.ndim) {
        @panic("Output has wrong number of dimensions");
    }

    var out_shape = reduceShape(in.getShape(), &[_]u64{dim}, keepdims);
    if (!std.mem.eql(u64, out.getShape(), out_shape.getSlice())) {
        @panic("Output has wrong shape");
    }

    var iterShape = DimArray.init(in.getShape());
    var reducedShape = DimArray.init(in.getShape());
    var d: u64 = 0;
    while (d < in.ndim) : (d += 1) {
        if (dim == d) {
            iterShape.array[d] = 1;
        } else {
            reducedShape.array[d] = 1;
        }
    }
    var in_it = PositionIterator.init(iterShape.getSlice());
    var out_it = out.createIterator();
    var out_buf = out.getBuffer(u64);
    while (in_it.next()) |pos| {
        const inner = in.narrowView(pos, reducedShape.getSlice());
        var inner_it = PositionIterator.init(inner.getShape());
        var out_offset = out_it.next().?;
        out_buf[out_offset] = switch (inner.buffer_union) {
            .u8 => argMaxBuffer(u8, inner, &inner_it, dim),
            .u64 => argMaxBuffer(u64, inner, &inner_it, dim),
            .i64 => argMaxBuffer(i64, inner, &inner_it, dim),
            .f32 => argMaxBuffer(f32, inner, &inner_it, dim),
            .f64 => argMaxBuffer(f64, inner, &inner_it, dim),
        };
    }
}

pub fn reduceArgMax(in: Array, out: Array, dim: u64) void {
    argMax(in, out, dim, false);
}

pub fn keepArgMax(in: Array, out: Array, dim: u64) void {
    argMax(in, out, dim, true);
}

pub fn reduceArgMaxAlloc(alc: *std.mem.Allocator, in: Array, dim: u64) !Array {
    var out_shape = reduceShape(in.getShape(), &[_]u64{dim}, false);
    var out = try zerosAlloc(alc, .u64, out_shape.getSlice());
    argMax(in, out, dim, false);
    return out;
}

pub fn keepArgMaxAlloc(alc: *std.mem.Allocator, in: Array, dim: u64) !Array {
    var out_shape = reduceShape(in.getShape(), &[_]u64{dim}, true);
    var out = try zerosAlloc(alc, .u64, out_shape.getSlice());
    argMax(in, out, dim, true);
    return out;
}

test "argmax" {
    var input_buf = [_]f32{ 1.0, 0.0, 0.0, 2.0, 1.0, 1.0 };
    const input = Array.fromBuffer(f32, &[_]u64{ 1, 3, 2 }, &input_buf);
    const output = try Array.allocWithValue(u64, std.testing.allocator, &[_]u64{ 1, 3 }, 0.0);
    defer output.release();
    reduceArgMax(input, output, 2);
    var expected_output_buf = [_]u64{ 0, 1, 0 };
    const expected_output = Array.fromBuffer(u64, &[_]u64{ 1, 3 }, &expected_output_buf);
    std.testing.expect(equal(output, expected_output));
    var output2 = try reduceArgMaxAlloc(std.testing.allocator, input, 2);
    defer output2.release();
    std.testing.expect(equal(output2, expected_output));
}

pub fn gather(in: Array, out: Array, dim: u64, index: Array) void {
    assertTypesAreTheSame(in, out);
    if (in.ndim != index.ndim) {
        @panic("Index must have same number of dimensions as input");
    }
    if (index.dtype != .u64) {
        @panic("Index must have u64 dtype");
    }
    for (in.getShape()) |s, d| {
        if (d != dim and index.getShape()[d] > s) {
            @panic("Index has invalid shape");
        }
    }
    assertShapesAreTheSame(out, index);

    var out_pos_it = PositionIterator.init(out.getShape());
    while (out_pos_it.next()) |out_pos| {
        var in_pos_shape = DimArray.init(out_pos);
        var in_pos = in_pos_shape.getSlice();
        in_pos[dim] = index.get(u64, out_pos);
        out.setValue(out_pos, in.getValue(in_pos));
    }
}

pub fn gatherAlloc(alc: *std.mem.Allocator, in: Array, dim: u64, index: Array) !Array {
    var out = try zerosAlloc(alc, in.dtype, index.getShape());
    gather(in, out, dim, index);
    return out;
}

test "gather" {
    const input = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0, 1.0);
    defer input.release();
    var index_data = [_]u64{ 0, 1, 2, 3, 2, 1 };
    const index = Array.fromBuffer(u64, &[_]u64{ 2, 3 }, &index_data);
    defer index.release();
    const index_expanded = index.reshapeView(&[_]u64{ 2, 3, 1 });
    const output = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 1 }, 0.0);
    defer output.release();
    gather(input, output, 2, index_expanded);
    var expected_output_data = [_]f32{ 0, 5, 10, 15, 18, 21 };
    const expected_output = Array.fromBuffer(f32, &[_]u64{ 2, 3, 1 }, &expected_output_data);
    std.testing.expect(equal(output, expected_output));
    const output2 = try gatherAlloc(std.testing.allocator, input, 2, index_expanded);
    defer output2.release();
    std.testing.expect(equal(output2, expected_output));
}

pub fn scatter(in: Array, out: Array, dim: u64, index: Array) void {
    assertTypesAreTheSame(in, out);
    if (out.ndim != index.ndim) {
        @panic("Index must have same number of dimensions as output");
    }
    for (out.getShape()) |s, d| {
        if (d != dim and index.getShape()[d] > s) {
            @panic("Index has invalid shape");
        }
    }
    assertShapesAreTheSame(in, index);

    var in_pos_it = PositionIterator.init(in.getShape());
    while (in_pos_it.next()) |in_pos| {
        var out_pos_shape = DimArray.init(in_pos);
        var out_pos = out_pos_shape.getSlice();
        out_pos[dim] = index.get(u64, in_pos);
        out.setValue(out_pos, in.getValue(in_pos));
    }
}

test "scatter" {
    const input = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 3, 1 }, 0.0, 1.0);
    defer input.release();
    var index_data = [_]u64{ 0, 1, 2, 3, 2, 1 };
    const index = Array.fromBuffer(u64, &[_]u64{ 2, 3 }, &index_data);
    defer index.release();
    const index_expanded = index.reshapeView(&[_]u64{ 2, 3, 1 });
    const output = try Array.allocWithValue(f32, std.testing.allocator, &[_]u64{ 2, 3, 4 }, 0.0);
    defer output.release();
    scatter(input, output, 2, index_expanded);
    var expected_output_data = [_]f32{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0 };
    const expected_output = Array.fromBuffer(f32, &[_]u64{ 2, 3, 4 }, &expected_output_data);
    std.testing.expect(equal(output, expected_output));
}

fn abs(comptime T: type, value: T) T {
    if (value > 0) {
        return value;
    } else {
        return -value;
    }
}

fn castToF64(comptime T: type, value: T) f64 {
    return switch (@typeInfo(T)) {
        .Int => @intToFloat(f64, value),
        else => value,
    };
}

fn allcloseBuffers(comptime T: type, x: Array, y: Array, rtol: f64, atol: f64) bool {
    var it_x = x.createIterator();
    var x_buf = x.getBuffer(T);
    var y_buf = y.getBuffer(T);
    var it_y = y.createIterator();
    while (it_x.next()) |x_offset| {
        var y_offset = it_y.next().?;
        var x_value = castToF64(T, x_buf[x_offset]);
        var y_value = castToF64(T, y_buf[y_offset]);
        var close = abs(f64, x_value - y_value) <= atol + rtol * abs(f64, y_value);
        if (!close) {
            return false;
        }
    }
    return true;
}

pub fn equal(x: Array, y: Array) bool {
    assertDimsAreTheSame(x, y);
    assertTypesAreTheSame(x, y);
    if (!std.mem.eql(u64, x.getShape(), y.getShape())) {
        return false;
    }
    return switch (x.dtype) {
        .u8 => allcloseBuffers(u8, x, y, 0.0, 0.0),
        .u64 => allcloseBuffers(u64, x, y, 0.0, 0.0),
        .i64 => allcloseBuffers(i64, x, y, 0.0, 0.0),
        .f32 => allcloseBuffers(f32, x, y, 0.0, 0.0),
        .f64 => allcloseBuffers(f64, x, y, 0.0, 0.0),
    };
}

pub fn allclose(x: Array, y: Array, rtol: f64, atol: f64) bool {
    assertDimsAreTheSame(x, y);
    assertTypesAreTheSame(x, y);
    if (!std.mem.eql(u64, x.getShape(), y.getShape())) {
        return false;
    }
    return switch (x.dtype) {
        .u8 => allcloseBuffers(u8, x, y, rtol, atol),
        .u64 => allcloseBuffers(u64, x, y, rtol, atol),
        .i64 => allcloseBuffers(i64, x, y, rtol, atol),
        .f32 => allcloseBuffers(f32, x, y, rtol, atol),
        .f64 => allcloseBuffers(f64, x, y, rtol, atol),
    };
}

fn expectContiguous(x: anytype, is_contiguous: bool) void {
    std.testing.expect(x.is_contiguous == is_contiguous);
    std.testing.expect(x.is_contiguous == checkContiguous(x));
}

fn dumpStruct(s: anytype) void {
    const type_info = @typeInfo(@TypeOf(s));
    comptime std.debug.assert(type_info == .Struct);
    var field_names: [type_info.Struct.fields.len][]const u8 = undefined;
    comptime {
        var i: comptime_int = 0;
        while (i < type_info.Struct.fields.len) {
            field_names[i] = type_info.Struct.fields[i].name;
            i += 1;
        }
    }
    comptime var i = 0;
    inline while (i < type_info.Struct.fields.len) {
        std.debug.print("name {} value {}\n", .{ type_info.Struct.fields[i].name, @field(s, type_info.Struct.fields[i].name) });
        i += 1;
    }
}

fn getValue(name: []const u8, names: [][]const u8, values: []i64) i64 {
    var i: u64 = 0;
    while (i < names.len) {
        if (std.mem.eql(u8, name, names[i])) {
            return values[i];
        }
        i += 1;
    }
    @panic("did not find value, this should never happen");
}

const MAX_TOKENS = 128;
const MAX_ITEMS = MAX_TOKENS;
const MAX_VALUES = 2 * MAX_TOKENS;
const MAX_OPERATIONS = MAX_TOKENS;
const MAX_FUNCTION_ARGS = 3;

const Operation = struct {
    function: Function,
    input_indices: [MAX_FUNCTION_ARGS]u64,
    output_index: u64,
};

fn functionToNumArgs(f: Function) u64 {
    return switch (f) {
        .plus => 2,
        .minus => 2,
        .uplus => 1,
        .uminus => 1,
        .times => 2,
        .mtimes => 2,
        .divide => 2,
        .mdivide => 2,
        .power => 2,
        .mpower => 2,
        .eq => 2,
        .gt => 2,
        .gte => 2,
        .lt => 2,
        .lte => 2,
        .transpose => 1,
        .ctranspose => 1,
        .f32 => 1,
        .detach => 1,
        .log => 1,
        .log2 => 1,
        .exp => 1,
        .max => 2,
        .reduce_sum => 2,
        .keep_sum => 2,
        .reduce_max => 2,
        .keep_max => 2,
        .reduce_mean => 2,
        .keep_mean => 2,
        .reduce_arg_max => 2,
        .keep_arg_max => 2,
        .gather => 3,
    };
}

fn functionToAutocastArgs(f: Function) []const u64 {
    return switch (f) {
        .plus => &[_]u64{ 0, 1 },
        .minus => &[_]u64{ 0, 1 },
        .uplus => &[_]u64{},
        .uminus => &[_]u64{},
        .times => &[_]u64{ 0, 1 },
        .mtimes => &[_]u64{ 0, 1 },
        .divide => &[_]u64{ 0, 1 },
        .mdivide => &[_]u64{ 0, 1 },
        .power => &[_]u64{ 0, 1 },
        .mpower => &[_]u64{ 0, 1 },
        .eq => &[_]u64{ 0, 1 },
        .gt => &[_]u64{ 0, 1 },
        .gte => &[_]u64{ 0, 1 },
        .lt => &[_]u64{ 0, 1 },
        .lte => &[_]u64{ 0, 1 },
        .transpose => &[_]u64{},
        .ctranspose => &[_]u64{},
        .f32 => &[_]u64{},
        .detach => &[_]u64{},
        .log => &[_]u64{},
        .log2 => &[_]u64{},
        .exp => &[_]u64{},
        .max => &[_]u64{ 0, 1 },
        .reduce_sum => &[_]u64{},
        .keep_sum => &[_]u64{},
        .reduce_max => &[_]u64{},
        .keep_max => &[_]u64{},
        .reduce_mean => &[_]u64{},
        .keep_mean => &[_]u64{},
        .reduce_arg_max => &[_]u64{},
        .keep_arg_max => &[_]u64{},
        .gather => &[_]u64{},
    };
}

const CompiledExpression = struct {
    operations: [MAX_OPERATIONS]Operation,
    operation_count: u64,
    numbers: [MAX_VALUES]Number,
    number_count: u64,
    index_count: u64,
    output_index: u64,
};

const TokenType = enum {
    operator,
    comma,
    number,
    identifier,
    leftParen,
    rightParen,
};

const Token = struct {
    typ: TokenType,
    val: []const u8,
};

const EOS = 0;

/// Scanner converts input string into tokens
const Scanner = struct {
    input: []const u8,
    start: u64 = 0, // start of current item
    pos: u64 = 0, // current pos within the input

    const Self = @This();

    fn init(input: []const u8) Self {
        return Self{ .input = input };
    }

    fn next(self: *Self) u8 {
        var c: u8 = undefined;
        if (self.pos >= self.input.len) {
            c = EOS;
        } else {
            c = self.input[self.pos];
        }
        self.pos += 1;
        return c;
    }

    fn backup(self: *Self) void {
        self.pos -= 1;
    }

    fn peek(self: *Self) u8 {
        var c = self.next();
        self.backup();
        return c;
    }

    fn ignore(self: *Self) void {
        self.start = self.pos;
    }

    fn accept(self: *Self, chars: []const u8) bool {
        var c = self.peek();
        if (contains(u8, chars, c)) {
            _ = self.next();
            return true;
        }
        return false;
    }

    fn acceptRun(self: *Self, chars: []const u8) bool {
        var result: bool = false;
        while (self.accept(chars)) {
            result = true;
        }
        return result;
    }

    fn lexChar(self: *Self) void {
        _ = self.next();
    }

    fn lexOperator(self: *Self) void {
        const operators = "*/^+-'";
        if (self.accept("><=")) {
            _ = self.accept("=");
        } else if (self.accept(".")) {
            _ = self.accept(operators);
        } else {
            _ = self.accept(operators);
        }
    }

    fn lexNumber(self: *Self) void {
        const digits = "0123456789";
        _ = self.acceptRun(digits);
        if (self.accept(".")) {
            _ = self.acceptRun(digits);
        }
        if (self.accept("eE")) {
            _ = self.accept("+-");
            _ = self.acceptRun(digits);
        }
    }

    fn lexIdentifier(self: *Self) void {
        if (self.accept("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")) {
            _ = self.acceptRun("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890_");
        }
    }

    fn nextLexeme(self: *Self) ?Token {
        // skip whitespace
        _ = self.acceptRun(" ");
        self.ignore();

        // look at next char
        var c = self.peek();

        if (c == EOS) {
            return null;
        }

        // extract the next item
        var typ: TokenType = undefined;
        if (c == ',' or c == '(' or c == ')') {
            // comma, this is a standalone item
            self.lexChar();
            if (c == ',') {
                typ = TokenType.comma;
            } else if (c == '(') {
                typ = TokenType.leftParen;
            } else if (c == ')') {
                typ = TokenType.rightParen;
            }
        } else if ('0' <= c and c <= '9') {
            self.lexNumber();
            typ = TokenType.number;
        } else if (c == '+' or c == '-' or c == '^' or c == '*' or c == '-' or c == '+' or c == '/' or c == '.' or c == '\'' or c == '>' or c == '<' or c == '=') {
            self.lexOperator();
            typ = TokenType.operator;
        } else if (('a' <= c and c <= 'z') or ('A' <= c and c <= 'Z')) {
            self.lexIdentifier();
            typ = TokenType.identifier;
        } else {
            @panic("unrecognized input");
        }
        var l = Token{ .typ = typ, .val = self.input[self.start..self.pos] };
        self.start = self.pos;
        return l;
    }

    fn getTokens(self: *Self, tokens: []Token) []Token {
        var count: u64 = 0;
        while (true) {
            if (count == tokens.len) {
                @panic("token buffer not large enough");
            }
            if (self.nextLexeme()) |l| {
                tokens[count] = l;
            } else {
                break;
            }
            count += 1;
        }
        return tokens[0..count];
    }
};

test "scanner" {
    var token_buffer: [MAX_TOKENS]Token = undefined;
    {
        var s = Scanner.init("(-aa +bb^2 .* cc -1.234 ) +");
        var tokens = s.getTokens(&token_buffer);
        var expected_tokens = [_][]const u8{ "(", "-", "aa", "+", "bb", "^", "2", ".*", "cc", "-", "1.234", ")", "+" };
        std.testing.expect(expected_tokens.len == tokens.len);
        for (tokens) |l, i| {
            std.testing.expect(std.mem.eql(u8, l.val, expected_tokens[i]));
        }
    }
}

const Function = enum {
    plus,
    minus,
    uplus,
    uminus,
    times,
    mtimes,
    divide,
    mdivide,
    power,
    mpower,
    transpose,
    ctranspose,
    detach,
    log,
    log2,
    exp,
    eq,
    gt,
    gte,
    lt,
    lte,
    f32,
    max,
    reduce_sum,
    keep_sum,
    reduce_max,
    keep_max,
    reduce_mean,
    keep_mean,
    reduce_arg_max,
    keep_arg_max,
    gather,
};

const Number = union(enum) {
    int: i64,
    float: f64,
};

const Parenthesis = enum {
    left,
    right,
};

const Value = struct {
    index: u64,
};

const Comma = struct {};

const Item = union(enum) {
    value: Value,
    number: Number,
    operator: Function,
    function: Function,
    paren: Parenthesis,
    comma: Comma,
};

/// Evaluator converts tokens into items
const Evaluator = struct {
    tokens: []Token,
    arg_names: [][]const u8,
    pos: u64 = 0,

    const Self = @This();

    fn init(tokens: []Token, arg_names: [][]const u8) Self {
        return Self{ .tokens = tokens, .arg_names = arg_names };
    }

    fn nextItem(self: *Self) ?Item {
        if (self.pos == self.tokens.len) {
            return null;
        }
        var prev_typ: ?TokenType = null;
        if (self.pos > 0) {
            prev_typ = self.tokens[self.pos - 1].typ;
        }
        var l = self.tokens[self.pos];
        self.pos += 1;
        return switch (l.typ) {
            .operator => blk: {
                if (std.mem.eql(u8, l.val, "+") or std.mem.eql(u8, l.val, ".+")) {
                    // differentiate uplus/uminus from plus/minus based on previous token
                    // operator: unary
                    // comma: unary
                    // number: binary
                    // identifier: binary
                    // left paren: unary
                    // right paren: binary
                    if (prev_typ == null or prev_typ.? == .operator or prev_typ.? == .comma or prev_typ.? == .leftParen) {
                        break :blk Item{ .operator = .uplus };
                    } else {
                        break :blk Item{ .operator = .plus };
                    }
                } else if (std.mem.eql(u8, l.val, "-") or std.mem.eql(u8, l.val, ".-")) {
                    if (prev_typ == null or prev_typ.? == .operator or prev_typ.? == .comma or prev_typ.? == .leftParen) {
                        break :blk Item{ .operator = .uminus };
                    } else {
                        break :blk Item{ .operator = .minus };
                    }
                } else if (std.mem.eql(u8, l.val, ".*")) {
                    break :blk Item{ .operator = .times };
                } else if (std.mem.eql(u8, l.val, "*")) {
                    break :blk Item{ .operator = .mtimes };
                } else if (std.mem.eql(u8, l.val, "./")) {
                    break :blk Item{ .operator = .divide };
                } else if (std.mem.eql(u8, l.val, "/")) {
                    break :blk Item{ .operator = .mdivide };
                } else if (std.mem.eql(u8, l.val, ".^")) {
                    break :blk Item{ .operator = .power };
                } else if (std.mem.eql(u8, l.val, "^")) {
                    break :blk Item{ .operator = .mpower };
                } else if (std.mem.eql(u8, l.val, "==")) {
                    break :blk Item{ .operator = .eq };
                } else if (std.mem.eql(u8, l.val, ">")) {
                    break :blk Item{ .operator = .gt };
                } else if (std.mem.eql(u8, l.val, ">=")) {
                    break :blk Item{ .operator = .gte };
                } else if (std.mem.eql(u8, l.val, "<")) {
                    break :blk Item{ .operator = .lt };
                } else if (std.mem.eql(u8, l.val, "<=")) {
                    break :blk Item{ .operator = .lte };
                } else if (std.mem.eql(u8, l.val, ".'")) {
                    break :blk Item{ .operator = .transpose };
                } else if (std.mem.eql(u8, l.val, "'")) {
                    break :blk Item{ .operator = .ctranspose };
                } else {
                    @panic("unrecognized operator");
                }
            },
            .comma => Item{ .comma = Comma{} },
            .number => blk: {
                if (contains(u8, l.val, '.') or contains(u8, l.val, 'e')) {
                    var float = std.fmt.parseFloat(f64, l.val) catch @panic("failed to parse float");
                    break :blk Item{ .number = Number{ .float = float } };
                } else {
                    var int = std.fmt.parseInt(i64, l.val, 10) catch @panic("failed to parse int");
                    break :blk Item{ .number = Number{ .int = int } };
                }
            },
            .identifier => blk: {
                if (std.mem.eql(u8, l.val, "plus")) {
                    break :blk Item{ .function = .plus };
                } else if (std.mem.eql(u8, l.val, "minus")) {
                    break :blk Item{ .function = .minus };
                } else if (std.mem.eql(u8, l.val, "uplus")) {
                    break :blk Item{ .function = .uplus };
                } else if (std.mem.eql(u8, l.val, "uminus")) {
                    break :blk Item{ .function = .uminus };
                } else if (std.mem.eql(u8, l.val, "times")) {
                    break :blk Item{ .function = .times };
                } else if (std.mem.eql(u8, l.val, "mtimes")) {
                    break :blk Item{ .function = .mtimes };
                } else if (std.mem.eql(u8, l.val, "divide")) {
                    break :blk Item{ .function = .divide };
                } else if (std.mem.eql(u8, l.val, "mdivide")) {
                    break :blk Item{ .function = .mdivide };
                } else if (std.mem.eql(u8, l.val, "power")) {
                    break :blk Item{ .function = .power };
                } else if (std.mem.eql(u8, l.val, "mpower")) {
                    break :blk Item{ .function = .mpower };
                } else if (std.mem.eql(u8, l.val, "eq")) {
                    break :blk Item{ .function = .eq };
                } else if (std.mem.eql(u8, l.val, "gt")) {
                    break :blk Item{ .function = .gt };
                } else if (std.mem.eql(u8, l.val, "gte")) {
                    break :blk Item{ .function = .gte };
                } else if (std.mem.eql(u8, l.val, "lt")) {
                    break :blk Item{ .function = .lt };
                } else if (std.mem.eql(u8, l.val, "lte")) {
                    break :blk Item{ .function = .lte };
                } else if (std.mem.eql(u8, l.val, "transpose")) {
                    break :blk Item{ .function = .transpose };
                } else if (std.mem.eql(u8, l.val, "ctranspose")) {
                    break :blk Item{ .function = .ctranspose };
                } else if (std.mem.eql(u8, l.val, "detach")) {
                    break :blk Item{ .function = .detach };
                } else if (std.mem.eql(u8, l.val, "f32")) {
                    break :blk Item{ .function = .f32 };
                } else if (std.mem.eql(u8, l.val, "log")) {
                    break :blk Item{ .function = .log };
                } else if (std.mem.eql(u8, l.val, "log2")) {
                    break :blk Item{ .function = .log2 };
                } else if (std.mem.eql(u8, l.val, "exp")) {
                    break :blk Item{ .function = .exp };
                } else if (std.mem.eql(u8, l.val, "max")) {
                    break :blk Item{ .function = .max };
                } else if (std.mem.eql(u8, l.val, "reduce_sum")) {
                    break :blk Item{ .function = .reduce_sum };
                } else if (std.mem.eql(u8, l.val, "keep_sum")) {
                    break :blk Item{ .function = .keep_sum };
                } else if (std.mem.eql(u8, l.val, "reduce_max")) {
                    break :blk Item{ .function = .reduce_max };
                } else if (std.mem.eql(u8, l.val, "keep_max")) {
                    break :blk Item{ .function = .keep_max };
                } else if (std.mem.eql(u8, l.val, "reduce_mean")) {
                    break :blk Item{ .function = .reduce_mean };
                } else if (std.mem.eql(u8, l.val, "keep_mean")) {
                    break :blk Item{ .function = .keep_mean };
                } else if (std.mem.eql(u8, l.val, "reduce_arg_max")) {
                    break :blk Item{ .function = .reduce_arg_max };
                } else if (std.mem.eql(u8, l.val, "keep_arg_max")) {
                    break :blk Item{ .function = .keep_arg_max };
                } else if (std.mem.eql(u8, l.val, "gather")) {
                    break :blk Item{ .function = .gather };
                } else {
                    // assume it's the name of an argument
                    var arg_index: u64 = 0;
                    while (arg_index < self.arg_names.len) {
                        if (std.mem.eql(u8, l.val, self.arg_names[arg_index])) {
                            break;
                        }
                        arg_index += 1;
                    }
                    if (arg_index == self.arg_names.len) {
                        @panic("invalid function name or argument missing");
                    }
                    break :blk Item{ .value = Value{ .index = arg_index } };
                }
            },
            .leftParen => Item{ .paren = .left },
            .rightParen => Item{ .paren = .right },
        };
    }

    fn getItems(self: *Self, items: []Item) []Item {
        var count: u64 = 0;
        while (true) {
            if (count == items.len) {
                @panic("Item buffer not large enough");
            }
            if (self.nextItem()) |item| {
                items[count] = item;
            } else {
                break;
            }
            count += 1;
        }
        return items[0..count];
    }
};

test "evaluator" {
    var token_buffer: [MAX_TOKENS]Token = undefined;
    var s = Scanner.init("(-aa +bb^2 .* cc -+1.234e-3 ) +");
    var tokens = s.getTokens(&token_buffer);
    var arg_names = [_][]const u8{ "aa", "bb", "cc" };
    var e = Evaluator.init(tokens, &arg_names);
    var item_buffer: [MAX_ITEMS]Item = undefined;
    var items = e.getItems(&item_buffer);
    var expected_items = [_]Item{
        Item{ .paren = .left },
        Item{ .operator = .uminus },
        Item{ .value = Value{ .index = 0 } },
        Item{ .operator = .plus },
        Item{ .value = Value{ .index = 1 } },
        Item{ .operator = .mpower },
        Item{ .number = Number{ .int = 2 } },
        Item{ .operator = .times },
        Item{ .value = Value{ .index = 2 } },
        Item{ .operator = .minus },
        Item{ .operator = .uplus },
        Item{ .number = Number{ .float = 1.234e-3 } },
        Item{ .paren = .right },
        Item{ .operator = .plus },
    };
    std.testing.expect(expected_items.len == items.len);

    for (items) |t, i| {
        std.testing.expect(itemsEqual(t, expected_items[i]));
    }
}

fn Stack(comptime T: type) type {
    return struct {
        buf: []T,
        len: u64 = 0,

        const Self = @This();

        fn init(buf: []T) Self {
            return Self{ .buf = buf };
        }

        fn push(self: *Self, item: T) void {
            self.buf[self.len] = item;
            self.len += 1;
        }

        fn pop(self: *Self) T {
            self.len -= 1;
            return self.buf[self.len];
        }

        fn top(self: *Self) T {
            if (self.len == 0) {
                @panic("attempted to get top of empty stack");
            }
            return self.buf[self.len - 1];
        }

        fn getSlice(self: *Self) []T {
            return self.buf[0..self.len];
        }
    };
}

fn operatorPrecedence(op: Function) u8 {
    // https://www.mathworks.com/help/matlab/matlab_prog/operator-precedence.html
    return switch (op) {
        .transpose, .ctranspose, .power, .mpower => 10,
        .uplus, .uminus => 8,
        .times, .mtimes, .divide, .mdivide => 7,
        .plus, .minus => 6,
        .eq, .gt, .gte, .lt, .lte => 4,
        .f32, .log, .log2, .exp, .reduce_sum, .keep_sum, .max, .reduce_max, .keep_max, .reduce_mean, .keep_mean, .reduce_arg_max, .keep_arg_max, .detach, .gather => @panic("not an operator"),
    };
}

const Associativity = enum {
    left,
    right,
};

fn operatorAssociativity(op: Function) Associativity {
    return switch (op) {
        .uminus, .uplus => .right,
        else => .left,
    };
}
/// infixToPostfix reorders items from infix to postfix
// https://en.wikipedia.org/wiki/Shunting-yard_algorithm
fn infixToPostfix(items_in: []Item, items_out_buf: []Item) []Item {
    var op_buf: [MAX_TOKENS]Item = undefined;
    var op_stack = Stack(Item).init(&op_buf);
    var output = Stack(Item).init(items_out_buf);

    var out_count: u64 = 0;
    // https://www.chris-j.co.uk/parsing.php
    for (items_in) |item| {
        switch (item) {
            // If the token is an operand, append it to the postfix output.
            .number, .value => {
                output.push(item);
            },
            // If the token is a unary postfix operator, append it to the postfix output.
            // We don't have any of these
            // If the token is a function token, push it on to the stack.
            .function => {
                op_stack.push(item);
            },
            .operator => |op| {
                // If the token is a unary prefix operator, push it on to the stack.
                if (functionToNumArgs(op) == 1) {
                    op_stack.push(item);
                } else {
                    // If the token is a binary operator A then
                    while (op_stack.len > 0) {
                        var top = op_stack.top();
                        if (top != .operator) {
                            break;
                        }
                        // If A is left-associative, while there is an operator B of higher or equal precidence than A at the top of the stack, pop B off the stack and append it to the output.
                        // If A is right-associative, while there is an operator B of higher precidence than A at the top of the stack, pop B off the stack and append it to the output.
                        var should_move = (operatorAssociativity(op) == .left and operatorPrecedence(top.operator) >= operatorPrecedence(op)) or (operatorAssociativity(op) == .right and operatorPrecedence(top.operator) > operatorPrecedence(op));
                        if (!should_move) {
                            break;
                        }
                        output.push(op_stack.pop());
                    }
                    // Push A onto the stack.
                    op_stack.push(item);
                }
            },
            // If the token is a function argument separator
            // Pop the top element off the stack and append it to the output, until the top element of the stack is an opening bracket
            .comma => {
                while (op_stack.len > 0) {
                    // Pop operators off the stack and append them to the output, until the operator at the top of the stack is a opening bracket.
                    var top = op_stack.top();
                    if (top == .paren and top.paren == .left) {
                        break;
                    }
                    output.push(op_stack.pop());
                }
            },
            .paren => |paren| {
                if (paren == .left) {
                    // If the token is an opening bracket, then push it onto the stack.
                    op_stack.push(item);
                } else {
                    // If the token is a closing bracket
                    var found_left_paren = false;
                    while (op_stack.len > 0) {
                        // Pop operators off the stack and append them to the output, until the operator at the top of the stack is a opening bracket.
                        var top = op_stack.top();
                        if (top == .paren and top.paren == .left) {
                            found_left_paren = true;
                            break;
                        }
                        output.push(op_stack.pop());
                    }
                    if (!found_left_paren) {
                        @panic("missing left parenthesis");
                    }
                    // Pop the opening bracket off the stack.
                    _ = op_stack.pop();
                    // If the token at the top of the stack is a function token, pop it and append it to the output.
                    if (op_stack.len > 0 and op_stack.top() == .function) {
                        output.push(op_stack.pop());
                    }
                }
            },
        }
    }
    while (op_stack.len > 0) {
        output.push(op_stack.pop());
    }
    return output.buf[0..output.len];
}

test "infixToPostfix" {
    var token_buffer: [MAX_TOKENS]Token = undefined;
    var s = Scanner.init("(aa + bb) .* cc");
    var tokens = s.getTokens(&token_buffer);
    var arg_names = [_][]const u8{ "aa", "bb", "cc" };
    var e = Evaluator.init(tokens, &arg_names);
    var item_buffer: [MAX_ITEMS]Item = undefined;
    var items = e.getItems(&item_buffer);
    var postfix_item_buffer: [MAX_ITEMS]Item = undefined;
    var postfix_items = infixToPostfix(items, &postfix_item_buffer);
    var expected_items = [_]Item{
        Item{ .value = Value{ .index = 0 } },
        Item{ .value = Value{ .index = 1 } },
        Item{ .operator = .plus },
        Item{ .value = Value{ .index = 2 } },
        Item{ .operator = .times },
    };
    std.testing.expect(expected_items.len == postfix_items.len);

    for (postfix_items) |t, i| {
        std.testing.expect(itemsEqual(t, expected_items[i]));
    }
}

fn itemsEqual(item1: Item, item2: Item) bool {
    return switch (item1) {
        .value => |value| value.index == item2.value.index,
        .number => |number| switch (number) {
            .int => |int| int == item2.number.int,
            .float => |float| float == item2.number.float,
        },
        .operator => |operator| operator == item2.operator,
        .function => |function| function == item2.function,
        .paren => |paren| paren == item2.paren,
        .comma => item2 == .comma,
    };
}

fn compileExpression(ex: []const u8, arg_names: [][]const u8) CompiledExpression {
    if (arg_names.len > MAX_VALUES) {
        @panic("too many arguments");
    }
    var token_buffer: [MAX_TOKENS]Token = undefined;
    var s = Scanner.init(ex);
    var tokens = s.getTokens(&token_buffer);
    var e = Evaluator.init(tokens, arg_names);
    var infix_item_buffer: [MAX_ITEMS]Item = undefined;
    var infix_items = e.getItems(&infix_item_buffer);
    var item_buffer: [MAX_ITEMS]Item = undefined;
    var items = infixToPostfix(infix_items, &item_buffer);

    if (items.len == 0) {
        @panic("no items found");
    }

    var index_buf: [MAX_VALUES]u64 = undefined;
    var index_stack = Stack(u64).init(&index_buf);
    var operations_buf: [MAX_OPERATIONS]Operation = undefined;
    var operations_stack = Stack(Operation).init(&operations_buf);
    var numbers_buf: [MAX_VALUES]Number = undefined;
    var numbers_stack = Stack(Number).init(&numbers_buf);

    var index_count: u64 = arg_names.len; // we reserve one value slot for each argument
    // also reserve a slot for each number literal
    // keep an ordered list of the numbers in numbers stack
    // keep a mapping from item index to index
    var number_item_index_to_index: [MAX_VALUES]u64 = [_]u64{0} ** MAX_VALUES;
    for (items) |item, item_index| {
        if (item == .number) {
            numbers_stack.push(item.number);
            number_item_index_to_index[item_index] = index_count;
            index_count += 1;
        }
    }

    for (items) |item, item_index| {
        switch (item) {
            .value => |value| {
                index_stack.push(value.index);
            },
            .number => |number| {
                index_stack.push(number_item_index_to_index[item_index]);
            },
            .operator, .function => |f| {
                // operator, pop operands from stack
                var num_args = functionToNumArgs(f);
                if (num_args > MAX_FUNCTION_ARGS) {
                    @panic("too many arguments for function");
                }
                var input_indices: [MAX_FUNCTION_ARGS]u64 = undefined;
                var i: u64 = 0;
                while (i < num_args) : (i += 1) {
                    // since we are popping the stack, if we had "a b -" we want a to be the first value
                    // and be to be the second one
                    input_indices[num_args - 1 - i] = index_stack.pop();
                }
                // put result on stack
                var output_value_index = index_count;
                index_count += 1;
                index_stack.push(output_value_index);
                // record the operation
                var operation = Operation{ .function = f, .input_indices = input_indices, .output_index = output_value_index };
                operations_stack.push(operation);
            },
            .paren, .comma => {
                @panic("parenthesis encountered after evaluation");
            },
        }
    }
    if (index_stack.len != 1) {
        @panic("did not process all values on stack");
    }
    return CompiledExpression{ .operations = operations_buf, .operation_count = operations_stack.len, .numbers = numbers_buf, .number_count = numbers_stack.len, .index_count = index_count, .output_index = index_stack.buf[0] };
}

/// Execute an expression using integer values, this is just for testing `compileExpression`
fn intExpr(comptime ex: []const u8, args: anytype) i64 {
    const type_info = @typeInfo(@TypeOf(args));
    const num_fields = type_info.Struct.fields.len;

    if (type_info != .Struct) {
        @compileError("must pass a struct to this function");
    }

    comptime var field_names: [num_fields][]const u8 = undefined;

    comptime {
        var i: comptime_int = 0;
        while (i < num_fields) : (i += 1) {
            field_names[i] = type_info.Struct.fields[i].name;
        }
    }

    @setEvalBranchQuota(10000);
    comptime var ce = compileExpression(ex, &field_names);

    if (ce.index_count == 0) {
        @compileError("expression used no values");
    }

    var values: [ce.index_count]i64 = undefined;

    // copy the fields into the values array
    comptime var field_index = 0;
    inline while (field_index < num_fields) : (field_index += 1) {
        values[field_index] = @field(args, type_info.Struct.fields[field_index].name);
    }

    // copy any literals from the expression to our values array after the fields
    comptime var number_index = 0;
    inline while (number_index < ce.number_count) : (number_index += 1) {
        values[num_fields + number_index] = ce.numbers[number_index].int;
    }

    // execute the operations, reading and writing values
    // this should be inlined at comptime but causes a compiler crash
    // comptime var op_index: u64 = 0;
    // inline while (op_index < ce.operation_count) : (op_index += 1) {
    var op_index: u64 = 0;
    while (op_index < ce.operation_count) : (op_index += 1) {
        var op = ce.operations[op_index];
        var x = values[op.input_indices[0]];
        // if this is not used, it will be the value at index 0
        // we assume that there's at least one value
        var y = values[op.input_indices[1]];
        var z = switch (op.function) {
            .plus => x + y,
            .minus => x - y,
            .uplus => x,
            .uminus => -x,
            .times => x * y,
            .mtimes => @panic("mtimes not supported"),
            .divide => @divTrunc(x, y),
            .mdivide => @panic("mdivide not supported"),
            .power => std.math.pow(i64, x, y),
            .mpower => @panic("mpower not supported"),
            .eq => boolToValue(i64, x == y),
            .gt => boolToValue(i64, x > y),
            .gte => boolToValue(i64, x >= y),
            .lt => boolToValue(i64, x < y),
            .lte => boolToValue(i64, x <= y),
            .transpose => x,
            .ctranspose => x,
            .f32 => @panic("f32 not supported"),
            .detach => @panic("keep_max not supported"),
            .log => @panic("ln not supported"),
            .log2 => @floatToInt(i64, std.math.log(f64, 2.0, @intToFloat(f64, x))),
            .exp => @floatToInt(i64, std.math.pow(f64, @intToFloat(f64, x), std.math.e)),
            .max => @panic("max not supported"),
            .reduce_sum => @panic("reduce_sum not supported"),
            .keep_sum => @panic("keep_sum not supported"),
            .reduce_max => @panic("reduce_max not supported"),
            .keep_max => @panic("keep_max not supported"),
            .reduce_mean => @panic("reduce_mean not supported"),
            .keep_mean => @panic("keep_mean not supported"),
            .reduce_arg_max => @panic("reduce_arg_max not supported"),
            .keep_arg_max => @panic("keep_arg_max not supported"),
            .gather => @panic("gather not supported"),
        };
        // std.debug.print("{} op {} => {}\n", .{x, y, z});
        values[op.output_index] = z;
    }
    return values[ce.output_index];
}

test "int_expr" {
    std.testing.expect(intExpr("-2 .^ 4", .{}) == -16);
    std.testing.expect(intExpr("power(uminus(2), 4)", .{}) == 16);
    std.testing.expect(intExpr("2 .^ 3 .^ 2", .{}) == 64);
    std.testing.expect(intExpr("2 + 3 .* 4", .{}) == 14);
    std.testing.expect(intExpr("(2 + 3) .* 4", .{}) == 20);
    std.testing.expect(intExpr("1 .* 2 + +(-3 + 4 .^ 5) .+ -(6 .- 7)", .{}) == 1024);
}

pub fn OpsTable(comptime T: type) type {
    const Error = error{OutOfMemory};
    const UnaryOpType = fn (*std.mem.Allocator, T) Error!T;
    const BinaryOpType = fn (*std.mem.Allocator, T, T) Error!T;
    const TernaryOpType = fn (*std.mem.Allocator, T, T, T) Error!T;
    const ScalarType = fn (*std.mem.Allocator, DType, f64) Error!T;
    const CastType = fn (*std.mem.Allocator, T, DType) Error!T;
    const DTypeType = fn (T) DType;
    return struct {
        plus: BinaryOpType,
        minus: BinaryOpType,
        uplus: UnaryOpType,
        uminus: UnaryOpType,
        times: BinaryOpType,
        mtimes: BinaryOpType,
        divide: BinaryOpType,
        mdivide: BinaryOpType,
        power: BinaryOpType,
        mpower: BinaryOpType,
        eq: BinaryOpType,
        gt: BinaryOpType,
        gte: BinaryOpType,
        lt: BinaryOpType,
        lte: BinaryOpType,
        transpose: UnaryOpType,
        ctranspose: UnaryOpType,
        scalar: ScalarType,
        cast: CastType,
        detach: UnaryOpType,
        log: UnaryOpType,
        log2: UnaryOpType,
        exp: UnaryOpType,
        // dims are passed as an array
        max: BinaryOpType,
        reduce_sum: BinaryOpType,
        keep_sum: BinaryOpType,
        reduce_max: BinaryOpType,
        keep_max: BinaryOpType,
        reduce_mean: BinaryOpType,
        keep_mean: BinaryOpType,
        reduce_arg_max: BinaryOpType,
        keep_arg_max: BinaryOpType,
        gather: TernaryOpType,
        get_dtype: DTypeType,
    };
}

pub fn binaryNotImplemented(alc: *std.mem.Allocator, x: Array, y: Array) !Array {
    @panic("operation not implemented");
}

pub fn unaryNotImplemented(alc: *std.mem.Allocator, x: Array) !Array {
    @panic("operation not implemented");
}

pub fn reduceSumExprAlloc(alc: *std.mem.Allocator, x: Array, dims: Array) !Array {
    var dims_cast = try castAlloc(alc, dims, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.getBuffer(u64);
    return try reduceSumAlloc(alc, x, dims_buf);
}

pub fn keepSumExprAlloc(alc: *std.mem.Allocator, x: Array, dims: Array) !Array {
    var dims_cast = try castAlloc(alc, dims, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.getBuffer(u64);
    return try keepSumAlloc(alc, x, dims_buf);
}

pub fn reduceMaxExprAlloc(alc: *std.mem.Allocator, x: Array, dims: Array) !Array {
    var dims_cast = try castAlloc(alc, dims, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.getBuffer(u64);
    return try reduceMaxAlloc(alc, x, dims_buf);
}

pub fn keepMaxExprAlloc(alc: *std.mem.Allocator, x: Array, dims: Array) !Array {
    var dims_cast = try castAlloc(alc, dims, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.getBuffer(u64);
    return try keepMaxAlloc(alc, x, dims_buf);
}

pub fn reduceMeanExprAlloc(alc: *std.mem.Allocator, x: Array, dims: Array) !Array {
    var dims_cast = try castAlloc(alc, dims, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.getBuffer(u64);
    return try reduceMeanAlloc(alc, x, dims_buf);
}

pub fn keepMeanExprAlloc(alc: *std.mem.Allocator, x: Array, dims: Array) !Array {
    var dims_cast = try castAlloc(alc, dims, .u64);
    defer dims_cast.release();
    var dims_buf = dims_cast.getBuffer(u64);
    return try keepMeanAlloc(alc, x, dims_buf);
}

pub fn reduceArgMaxExprAlloc(alc: *std.mem.Allocator, x: Array, dim: Array) !Array {
    var dim_cast = try castAlloc(alc, dim, .u64);
    defer dim_cast.release();
    return try reduceArgMaxAlloc(alc, x, dim_cast.getItem(u64));
}

pub fn keepArgMaxExprAlloc(alc: *std.mem.Allocator, x: Array, dim: Array) !Array {
    var dim_cast = try castAlloc(alc, dim, .u64);
    defer dim_cast.release();
    return try keepArgMaxAlloc(alc, x, dim_cast.getItem(u64));
}

pub fn gatherExprAlloc(alc: *std.mem.Allocator, x: Array, dim: Array, index: Array) !Array {
    var dim_cast = try castAlloc(alc, dim, .u64);
    defer dim_cast.release();
    var index_cast = try castAlloc(alc, index, .u64);
    defer index_cast.release();
    return try gatherAlloc(alc, x, dim_cast.getItem(u64), index_cast);
}

fn getDType(a: Array) DType {
    return a.dtype;
}

pub fn expr(alc: *std.mem.Allocator, comptime ex: []const u8, args: anytype) !Array {
    comptime var opsTable = OpsTable(Array){
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
        .eq = eqAlloc,
        .gt = gtAlloc,
        .gte = gteAlloc,
        .lt = ltAlloc,
        .lte = lteAlloc,
        .transpose = transposeAlloc,
        .ctranspose = transposeAlloc,
        .scalar = scalarAlloc,
        .cast = castAlloc,
        .detach = unaryNotImplemented,
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
        .reduce_arg_max = reduceArgMaxExprAlloc,
        .keep_arg_max = keepArgMaxExprAlloc,
        .gather = gatherExprAlloc,
        .get_dtype = getDType,
    };
    return try genericExpr(Array, opsTable, alc, ex, args);
}

/// Execute an expression using array-like values
pub fn genericExpr(comptime T: type, comptime opsTable: OpsTable(T), alc: *std.mem.Allocator, comptime ex: []const u8, args: anytype) !T {
    const type_info = @typeInfo(@TypeOf(args));
    const num_fields = type_info.Struct.fields.len;

    if (type_info != .Struct) {
        @compileError("must pass a struct to this function");
    }

    comptime var field_names: [num_fields][]const u8 = undefined;

    comptime {
        var i: comptime_int = 0;
        while (i < num_fields) : (i += 1) {
            field_names[i] = type_info.Struct.fields[i].name;
        }
    }

    @setEvalBranchQuota(10000);
    comptime var ce = compileExpression(ex, &field_names);

    if (ce.index_count == 0) {
        @compileError("expression used no values");
    }

    var values: [ce.index_count]T = undefined;
    var allocated_scalar = [_]bool{false} ** num_fields;

    // copy the fields into the values array
    comptime var field_index = 0;
    inline while (field_index < num_fields) : (field_index += 1) {
        const field_type = type_info.Struct.fields[field_index].field_type;
        const value = @field(args, type_info.Struct.fields[field_index].name);
        values[field_index] = switch (field_type) {
            T => value,
            comptime_int => blk: {
                allocated_scalar[field_index] = true;
                break :blk try opsTable.scalar(alc, defaultIntDType, value);
            },
            comptime_float => blk: {
                allocated_scalar[field_index] = true;
                break :blk try opsTable.scalar(alc, defaultFloatDType, value);
            },
            i64, u64 => blk: {
                allocated_scalar[field_index] = true;
                break :blk try opsTable.scalar(alc, typeToDType(field_type), @intToFloat(f64, value));
            },
            f32, f64 => blk: {
                allocated_scalar[field_index] = true;
                break :blk try opsTable.scalar(alc, typeToDType(field_type), @floatCast(f64, value));
            },
            else => std.debug.panic("Unsupported type {}", .{@typeName(field_type)}),
        };
    }

    // copy any literals from the expression to our values array after the fields
    comptime var number_index = 0;
    inline while (number_index < ce.number_count) : (number_index += 1) {
        var arr = switch (ce.numbers[number_index]) {
            // these should probably use the minimum type that represents the value
            // to avoid accidentally casting arguments to larger types
            .int => |int| try opsTable.scalar(alc, defaultIntDType, int),
            .float => |float| try opsTable.scalar(alc, defaultFloatDType, float),
        };
        var value_index = num_fields + number_index;
        values[value_index] = arr;
    }

    // execute the operations, reading and writing values
    // this should be inlined at comptime but causes a compiler crash
    var op_args: [MAX_FUNCTION_ARGS]T = undefined;
    // comptime var op_index: u64 = 0;
    // inline while (op_index < ce.operation_count) : (op_index += 1) {
    var op_index: u64 = 0;
    while (op_index < ce.operation_count) : (op_index += 1) {
        var op = ce.operations[op_index];
        var num_args = functionToNumArgs(op.function);
        var should_release: [MAX_FUNCTION_ARGS]bool = [_]bool{false} ** MAX_FUNCTION_ARGS;
        // automatic casting rules
        // for functions that take two arguments
        //      cast_dtype = max(dtype[0], dtype[1])
        // where the dtypes are ordered: all ints, then all floats in order of bit width
        // some operations are handled specially:
        //    divide()
        //      if both args are integer types, they will be cast to the default float type
        {
            var arg_index: u64 = 0;
            while (arg_index < num_args) : (arg_index += 1) {
                var value_index = op.input_indices[arg_index];
                var value = values[value_index];
                // we shouldn't release the original args to this function, but we should
                // release any literals or calculated values since those are only used once
                if (value_index >= num_fields) {
                    should_release[arg_index] = true;
                }
                op_args[arg_index] = value;
            }
        }
        // do any autocasting
        var autocast_args = functionToAutocastArgs(op.function);
        if (autocast_args.len > 0) {
            var cast_dtype = DType.u64;
            for (autocast_args) |arg_index| {
                cast_dtype = dtypeMax(cast_dtype, opsTable.get_dtype(op_args[autocast_args[arg_index]]));
            }
            if (dtypeIsInteger(cast_dtype) and op.function == .divide) {
                cast_dtype = defaultFloatDType;
            }
            var arg_index: u64 = 0;
            while (arg_index < num_args) : (arg_index += 1) {
                if (opsTable.get_dtype(op_args[arg_index]) != cast_dtype) {
                    var cast_value = try opsTable.cast(alc, op_args[arg_index], cast_dtype);
                    if (should_release[arg_index]) {
                        op_args[arg_index].release();
                    }
                    op_args[arg_index] = cast_value;
                    // since we allocate this cast, we should free it when we are done with this operation
                    should_release[arg_index] = true;
                }
            }
        }
        var out = switch (op.function) {
            .plus => try opsTable.plus(alc, op_args[0], op_args[1]),
            .minus => try opsTable.minus(alc, op_args[0], op_args[1]),
            .uplus => try opsTable.uplus(alc, op_args[0]),
            .uminus => try opsTable.uminus(alc, op_args[0]),
            .times => try opsTable.times(alc, op_args[0], op_args[1]),
            .mtimes => try opsTable.mtimes(alc, op_args[0], op_args[1]),
            .divide => try opsTable.divide(alc, op_args[0], op_args[1]),
            .mdivide => try opsTable.mdivide(alc, op_args[0], op_args[1]),
            .power => try opsTable.power(alc, op_args[0], op_args[1]),
            .mpower => try opsTable.mpower(alc, op_args[0], op_args[1]),
            .eq => try opsTable.eq(alc, op_args[0], op_args[1]),
            .gt => try opsTable.gt(alc, op_args[0], op_args[1]),
            .gte => try opsTable.gte(alc, op_args[0], op_args[1]),
            .lt => try opsTable.lt(alc, op_args[0], op_args[1]),
            .lte => try opsTable.lte(alc, op_args[0], op_args[1]),
            .transpose => try opsTable.transpose(alc, op_args[0]),
            .ctranspose => try opsTable.ctranspose(alc, op_args[0]),
            .detach => try opsTable.detach(alc, op_args[0]),
            .log => try opsTable.log(alc, op_args[0]),
            .log2 => try opsTable.log2(alc, op_args[0]),
            .exp => try opsTable.exp(alc, op_args[0]),
            .f32 => try opsTable.cast(alc, op_args[0], DType.f32),
            .max => try opsTable.max(alc, op_args[0], op_args[1]),
            .reduce_sum => try opsTable.reduce_sum(alc, op_args[0], op_args[1]),
            .keep_sum => try opsTable.keep_sum(alc, op_args[0], op_args[1]),
            .reduce_max => try opsTable.reduce_max(alc, op_args[0], op_args[1]),
            .keep_max => try opsTable.keep_max(alc, op_args[0], op_args[1]),
            .reduce_mean => try opsTable.reduce_mean(alc, op_args[0], op_args[1]),
            .keep_mean => try opsTable.keep_mean(alc, op_args[0], op_args[1]),
            .reduce_arg_max => try opsTable.reduce_arg_max(alc, op_args[0], op_args[1]),
            .keep_arg_max => try opsTable.keep_arg_max(alc, op_args[0], op_args[1]),
            .gather => try opsTable.gather(alc, op_args[0], op_args[1], op_args[2]),
        };

        {
            var arg_index: u64 = 0;
            while (arg_index < num_args) : (arg_index += 1) {
                if (should_release[arg_index]) {
                    op_args[arg_index].release();
                }
            }
        }
        values[op.output_index] = out;
    }
    // free any scalars we allocated from the struct fields
    comptime var index = 0;
    inline while (index < allocated_scalar.len) : (index += 1) {
        if (allocated_scalar[index]) {
            values[index].release();
        }
    }
    return values[ce.output_index];
}

test "expr" {
    {
        const a = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0, 1.0);
        defer a.release();
        const b = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 1, 4, 1 }, 1.0, 1.0);
        defer b.release();
        var c = try expr(std.testing.allocator, "-1 + a + b + 1", .{ .a = a, .b = b });
        defer c.release();
        var d = try plusAlloc(std.testing.allocator, a, b);
        defer d.release();
        std.testing.expect(equal(c, d));
    }
    {
        const a = try Array.allocWithRange(i64, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0, 1.0);
        defer a.release();
        const b = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 1, 4, 1 }, 1.0, 1.0);
        defer b.release();
        var c = try expr(std.testing.allocator, "-1 + f32(a) + b + 1", .{ .a = a, .b = b });
        defer c.release();
        const d = try Array.allocWithRange(f32, std.testing.allocator, &[_]u64{ 2, 1, 3 }, 1.0, 1.0);
        defer d.release();
        var e = try plusAlloc(std.testing.allocator, d, b);
        defer e.release();
        std.testing.expect(equal(c, e));
    }
    {
        var output = try expr(std.testing.allocator, "a + b", .{ .a = 1, .b = 2 });
        defer output.release();
        const expected_output = try Array.allocWithValue(i64, std.testing.allocator, &[_]u64{}, 3);
        defer expected_output.release();
        std.testing.expect(equal(output, expected_output));
    }
}

// test "debug_expr" {
//     var token_buffer: [MAX_TOKENS]Token = undefined;
//     var s = Scanner.init("reduce_sum(-x, dims)/size");
//     var tokens = s.getTokens(&token_buffer);
//     for (tokens) |t, i| {
//         std.debug.print("token {}\n", .{t});
//     }
//     var arg_names = [_][]const u8{"x", "dims", "size"};
//     var e = Evaluator.init(tokens, &arg_names);
//     var item_buffer: [MAX_ITEMS]Item = undefined;
//     var items = e.getItems(&item_buffer);
//     for (items) |t, i| {
//         std.debug.print("infix item {}\n", .{t});
//     }
//     var postfix_item_buffer: [MAX_ITEMS]Item = undefined;
//     var postfix_items = infixToPostfix(items, &postfix_item_buffer);
//     for (postfix_items) |t, i| {
//         std.debug.print("item {}\n", .{t});
//     }
// }
