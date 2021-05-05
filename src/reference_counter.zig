// Keep a reference count that is incremented and decremented atomically

const std = @import("std");
const builtin = @import("builtin");


pub const ReferenceCounter = struct {
    ref_count: usize,

    const Self = @This();

    pub fn init() Self {
        return Self{ .ref_count = 1 };
    }

    pub fn increment(self: *Self) void {
        // atomically increment the reference count
        var ref_count = self.ref_count;
        while (true) {
            if (ref_count == 0) {
                // reference count could be zero briefly, but if it's zero, then it's about to be deallocated
                // and then it can have any value
                @panic("reference count is zero");
            }
            var new_ref_count = ref_count + 1;
            var result = @cmpxchgWeak(usize, &self.ref_count, ref_count, new_ref_count, builtin.AtomicOrder.Monotonic, builtin.AtomicOrder.Monotonic);
            if (result == null) {
                break;
            }
            ref_count = result.?;
        }
    }

    pub fn decrement(self: *Self) bool {
        // atomically decrement the ref count
        // return true if the ref count hit zero
        var ref_count = self.ref_count;
        while (true) {
            var new_ref_count = ref_count - 1;
            var result = @cmpxchgWeak(usize, &self.ref_count, ref_count, new_ref_count, builtin.AtomicOrder.Monotonic, builtin.AtomicOrder.Monotonic);
            if (result == null) {
                // the exchange was a success
                return new_ref_count == 0;
            }
            ref_count = result.?;
        }
    }
};