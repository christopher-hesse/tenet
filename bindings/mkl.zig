// MKL bindings for fast CPU operations

const std = @import("std");

const C = @cImport({
    @cInclude("mkl.h");
});

pub fn cblas_sgemm(a: *f32, b: *f32, c: *f32, lda: u64, ldb: u64, ldc: u64, m: u64, n: u64, k: u64, alpha: f32, beta: f32) void {
    var lda_int = @intCast(C.MKL_INT, lda);
    var ldb_int = @intCast(C.MKL_INT, ldb);
    var ldc_int = @intCast(C.MKL_INT, ldc);
    var m_int = @intCast(C.MKL_INT, m);
    var n_int = @intCast(C.MKL_INT, n);
    var k_int = @intCast(C.MKL_INT, k);
    var layout: C.CBLAS_LAYOUT = C.CBLAS_LAYOUT.CblasRowMajor;
    var transa: C.CBLAS_TRANSPOSE = C.CBLAS_TRANSPOSE.CblasNoTrans;
    var transb: C.CBLAS_TRANSPOSE = C.CBLAS_TRANSPOSE.CblasNoTrans;
    C.cblas_sgemm(
        layout,
        transa,
        transb,
        m_int,
        n_int,
        k_int,
        alpha,
        a,
        lda_int,
        b,
        ldb_int,
        beta,
        c,
        ldc_int,
    );
}

pub fn cblas_dgemm(a: *f64, b: *f64, c: *f64, lda: u64, ldb: u64, ldc: u64, m: u64, n: u64, k: u64, alpha: f64, beta: f64) void {
    var lda_int = @intCast(C.MKL_INT, lda);
    var ldb_int = @intCast(C.MKL_INT, ldb);
    var ldc_int = @intCast(C.MKL_INT, ldc);
    var m_int = @intCast(C.MKL_INT, m);
    var n_int = @intCast(C.MKL_INT, n);
    var k_int = @intCast(C.MKL_INT, k);
    var layout: C.CBLAS_LAYOUT = C.CBLAS_LAYOUT.CblasRowMajor;
    var transa: C.CBLAS_TRANSPOSE = C.CBLAS_TRANSPOSE.CblasNoTrans;
    var transb: C.CBLAS_TRANSPOSE = C.CBLAS_TRANSPOSE.CblasNoTrans;
    C.cblas_dgemm(
        layout,
        transa,
        transb,
        m_int,
        n_int,
        k_int,
        alpha,
        a,
        lda_int,
        b,
        ldb_int,
        beta,
        c,
        ldc_int,
    );
}
