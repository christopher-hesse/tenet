// MNIST example based on:
//   https://github.com/milindmalshe/Fully-Connected-Neural-Network-MNIST-Classification-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
// Download dataset from http://yann.lecun.com/exdb/mnist/
// Run with `zig build -Drelease-fast run -- <path to folder with mnist files>`
// If you have MKL installed from https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html
//   you can run with `zig build -Drelease-fast -Duse-mkl run -- <data path>` to use a much faster matrix multiply operation

const std = @import("std");

const tenet = @import("tenet.zig");
const Array = tenet.Array;
const Tensor = tenet.Tensor;

fn readIdx(comptime T: type, alc: *std.mem.Allocator, dirpath: []const u8, filename: []const u8, magic_number: i32, comptime num_dims: comptime_int) !Tensor {
    // check for the already extracted file, if it is missing, extract it from the provided compressed file path
    if (!std.mem.eql(u8, filename[filename.len-3..], ".gz")) {
        std.debug.panic("Filename should end in .gz: {}", .{filename});
    }
    const extracted_filename = filename[0..filename.len - 3];
    var f : std.fs.File = undefined;
    var dir = try std.fs.cwd().openDir(dirpath, .{});
    defer dir.close();
    if (dir.openFile(extracted_filename, std.fs.File.OpenFlags{ .read = true })) |ok| {
        f = ok;
    } else |err| switch (err) {
        error.FileNotFound => {
            // extract the file
            {
                var fw = try dir.createFile(extracted_filename, std.fs.File.CreateFlags{});
                defer fw.close();
                var fr = try dir.openFile(filename, std.fs.File.OpenFlags{ .read = true });
                defer fr.close();
                var s = try std.compress.gzip.gzipStream(alc, fr.reader());
                defer s.deinit();
                var buf : [4096]u8 = undefined;
                var total_nbytes : u64 = 0;
                while (true) {
                    var nbytes = try s.read(&buf);
                    if (nbytes == 0) {
                        break;
                    }
                    try fw.writeAll(buf[0..nbytes]);
                    total_nbytes += nbytes;
                }
            }
            // open the extracted file
            f = try dir.openFile(extracted_filename, std.fs.File.OpenFlags{ .read = true });
        },
        else => {
            std.debug.panic("Failed to open file {}", .{err});
        },
    }
    defer f.close();
    var r = f.reader();

    var num = try r.readIntBig(i32);
    if (num != magic_number) {
        return error.InvalidFile;
    }
    var shape = [_]u64{0} ** num_dims;
    for (shape) |_, i| {
        shape[i] = @intCast(u64, try r.readIntBig(i32));
    }
    // create array, read into it
    var result = try Tensor.allocWithValue(T, alc, &shape, 0, tenet.tensor.NO_FLAGS);
    errdefer result.release();
    var data_buf = result.data.getBuffer(T);
    var nbytes = try r.read(data_buf);
    if (nbytes != data_buf.len) {
        return error.InvalidFile;
    }
    return result;
}

fn loadImageData(alc: *std.mem.Allocator, dirpath: []const u8, filename: []const u8) !Tensor {
    std.debug.print("reading {}/{}\n", .{dirpath, filename});
    return readIdx(u8, alc, dirpath, filename, 2051, 3);
}

fn loadLabelData(alc: *std.mem.Allocator, dirpath: []const u8, filename: []const u8) !Tensor {
    std.debug.print("reading {}/{}\n", .{dirpath, filename});
    return readIdx(u8, alc, dirpath, filename, 2049, 1);
}

fn preprocessImages(alc: *std.mem.Allocator, images: Tensor) !Tensor {
    return try tenet.tensor.expr(alc, "f32(images) ./ 255.0", .{.images=images});
}

const Model = struct {
    mlp: tenet.module.MLP,

    const Self = @This();

    fn init(alc: *std.mem.Allocator, rng: *std.rand.Random, input_size: u64, hidden_size: u64, output_size: u64) !Self {
        return Self{.mlp=try tenet.module.MLP.init(alc, rng, input_size, hidden_size, output_size)};
    }

    fn collectParameters(self: Self, pc: tenet.module.ParameterCollector) !void {
        try pc.collectParameters(self, "mlp");
    }

    fn forward(self: *Self, alc: *std.mem.Allocator, x: Tensor) !Tensor {
        var logits = try self.mlp.forward(alc, x);
        defer logits.release();
        return try tenet.funcs.logSoftmax(alc, logits, &[_]u64{1});
    }

    fn deinit(self: *Self) void {
        self.mlp.deinit();
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var alc = &gpa.allocator;

    var args = try std.process.argsAlloc(alc);
    defer std.process.argsFree(alc, args);

    if (args.len != 2) {
        @panic("Incorrect number of args, must provide path to MNIST data");
    }

    var dataset_path = args[1];

    const batch_size : u64 = 100;
    const in_features : u64 = 28*28;
    const hidden_features : u64 = 500;
    const out_features : u64 = 10;
    const num_epochs : u64 = 5;
    const learning_rate : f32 = 0.001;

    var gen = std.rand.Xoroshiro128.init(0);
    var model = try Model.init(alc, &gen.random, in_features, hidden_features, out_features);
    defer model.deinit();
    
    var train_images_raw = try loadImageData(alc, dataset_path, "train-images-idx3-ubyte.gz");
    defer train_images_raw.release();
    var train_images = try preprocessImages(alc, train_images_raw);
    defer train_images.release();
    var train_labels = try loadLabelData(alc, dataset_path, "train-labels-idx1-ubyte.gz");
    defer train_labels.release();
    
    var test_images_raw = try loadImageData(alc, dataset_path, "t10k-images-idx3-ubyte.gz");
    defer test_images_raw.release();
    var test_images = try preprocessImages(alc, test_images_raw);
    defer test_images.release();
    var test_labels = try loadLabelData(alc, dataset_path, "t10k-labels-idx1-ubyte.gz");
    defer test_labels.release();

    var pc = try tenet.module.ParameterCollector.init(alc);
    defer pc.deinit();
    try model.collectParameters(pc);

    var opt = try tenet.optim.SGD.init(alc, pc.getParameters(), 0.9);
    // var opt = try tenet.optim.Adam.init(alc, pc.getParameters(), 0.9, 0.999, 1e-8);
    defer opt.deinit();

    var epoch : u64 = 0;

    const num_train_batches = @divTrunc(train_images.data.getShape()[0], batch_size);
    const num_test_batches = @divTrunc(test_images.data.getShape()[0], batch_size);

    while (epoch < num_epochs) : (epoch += 1) {
        std.debug.print("epoch {}\n", .{epoch});
        var batch_index : u64 = 0;
        var start = std.time.milliTimestamp();
        var image_count : u64 = 0;


        while (batch_index < num_train_batches) : (batch_index += 1) {
            var input = train_images.narrowView(&[_]u64{batch_size * batch_index, 0, 0}, &[_]u64{batch_size, 28, 28});
            var input_flat = input.reshapeView(&[_]u64{batch_size, in_features});
            var labels = train_labels.narrowView(&[_]u64{batch_size * batch_index}, &[_]u64{batch_size});
            var logprobs = try model.forward(alc, input_flat);
            defer logprobs.release();
            var loss = try tenet.funcs.nllLoss(alc, logprobs, labels);
            defer loss.release();

            try opt.zeroGrad();
            try tenet.tensor.backwardScalarAlloc(alc, loss);
            try opt.step(learning_rate);

            var end = std.time.milliTimestamp();
            image_count += batch_size;
            if (batch_index % 100 == 0) {
                var rate = @intToFloat(f32, image_count) / ((@intToFloat(f32, end - start) / 1000));
                std.debug.print("train step batch_index {} num_train_batches {} loss {} rate {}\n", .{batch_index, num_train_batches, loss.data.getItem(f32), @floatToInt(u64, rate)});
            }
        }

        batch_index = 0;
        var total : u64 = 0;
        var correct : u64 = 0;
        while (batch_index < num_test_batches) : (batch_index += 1) {
            var input = test_images.narrowView(&[_]u64{batch_size * batch_index, 0, 0}, &[_]u64{batch_size, 28, 28});
            var input_flat = input.reshapeView(&[_]u64{batch_size, in_features});
            var logprobs = try model.forward(alc, input_flat);
            defer logprobs.release();
            var labels = test_labels.narrowView(&[_]u64{batch_size * batch_index}, &[_]u64{batch_size});
            var dims = [_]u64{0};
            var correct_count = try tenet.array.expr(alc, "reduce_sum(reduce_arg_max(logprobs, 1) == labels, dims)", .{.logprobs=logprobs.data, .labels=labels.data, .dims=Array.flatFromBuffer(u64, &dims)});
            defer correct_count.release();
            total += input.data.getShape()[0];
            correct += correct_count.getItem(u64);
        }
        var accuracy : f32 = @intToFloat(f32, correct) / @intToFloat(f32, total);
        std.debug.print("test step correct {} total {} accuracy {}%\n", .{correct, total, @floatToInt(u64, accuracy * 100)});
    }
}

