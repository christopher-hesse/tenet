const std = @import("std");

const mkl_path = "C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\2021.2.0\\";
const tbb_path = "C:\\Program Files (x86)\\Intel\\oneAPI\\tbb\\2021.2.0\\";

fn linkMKL(exe : *std.build.LibExeObjStep) void {
    exe.addIncludeDir(mkl_path ++ "include");
    exe.addLibPath(mkl_path ++ "lib\\intel64");
    exe.linkSystemLibrary("mkl_core");
    exe.linkSystemLibrary("mkl_tbb_thread");
    exe.linkSystemLibrary("mkl_intel_thread");
    exe.linkSystemLibrary("mkl_intel_lp64");
    exe.addLibPath(tbb_path ++ "lib\\intel64\\vc14");
    // even after doing this, tbb12 still seems to be dynamically linked
    // exe.linkSystemLibrary("tbb12");
    exe.linkLibC();
}


pub fn build(b: *std.build.Builder) void {
    const link_mkl = b.option(bool, "link-mkl", "Link the MKL library") orelse false;

    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("main", "src/main.zig");
    exe.setTarget(target);
    exe.setBuildMode(mode);
    if (link_mkl) {
        linkMKL(exe);
    }
    exe.install();

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
    
    const test_step = b.step("test", "Run library tests");

    for ([_][]const u8{
        "src/tensor.zig", // for some reason this also runs array.zig's tests
        // "src/array.zig",
        "src/mkl.zig",
    }) |test_file| {
        const tests = b.addTest(test_file);
        tests.setTarget(target);
        tests.setBuildMode(mode);
        if (link_mkl) {
            linkMKL(tests);
        }
        test_step.dependOn(&tests.step);
    }
}
