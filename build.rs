// Build the C++ coder and the vendored huff0 into a static library the crate links against.
// There is no system dependency: everything compiled here ships in the crate.
fn main() {
    let mut b = cc::Build::new();
    b.cpp(true)
        .std("c++17")
        .include(".")
        .include("third_party/fse")
        .file("plane_entropy.cpp")
        .opt_level(3)
        .warnings(false);
    if cfg!(feature = "no-simd") {
        b.define("PE_NO_SIMD", None);
    }
    b.compile("plane_entropy_cpp");

    let mut c = cc::Build::new();
    c.include("third_party/fse").opt_level(3).warnings(false);
    for f in [
        "huf_compress.c",
        "huf_decompress.c",
        "fse_compress.c",
        "fse_decompress.c",
        "entropy_common.c",
        "hist.c",
        "debug.c",
    ] {
        c.file(format!("third_party/fse/{f}"));
    }
    c.compile("plane_entropy_fse");

    println!("cargo:rerun-if-changed=plane_entropy.hpp");
    println!("cargo:rerun-if-changed=plane_entropy.cpp");
    println!("cargo:rerun-if-changed=third_party/fse");
}
