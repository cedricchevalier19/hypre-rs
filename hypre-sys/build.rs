extern crate bindgen;

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

fn env_inner(name: &str) -> Option<OsString> {
    let var = env::var_os(name);
    println!("cargo:rerun-if-env-changed={}", name);

    match var {
        Some(ref v) => println!("{} = {}", name, v.to_string_lossy()),
        None => println!("{} unset", name),
    }

    var
}

fn env(name: &str) -> Option<OsString> {
    let prefix = env::var("TARGET").unwrap().to_uppercase().replace('-', "_");
    let prefixed = format!("{}_{}", prefix, name);
    env_inner(&prefixed).or_else(|| env_inner(name))
}

pub fn find_hypre() -> (Vec<PathBuf>, PathBuf) {
    let lib_dir = env("HYPRE_LIB_DIR").map(PathBuf::from);
    let include_dir = env("HYPRE_INCLUDE_DIR").map(PathBuf::from);

    match (lib_dir, include_dir) {
        (Some(lib_dir), Some(include_dir)) => (vec![lib_dir], include_dir),
        (lib_dir, include_dir) => {
            let hypre_dir = env("HYPRE_DIR").unwrap_or_else(|| "/usr".into());
            let hypre_dir = Path::new(&hypre_dir);
            let lib_dir = lib_dir.map(|d| vec![d]).unwrap_or_else(|| {
                let mut lib_dirs = vec![];
                // check for both it and lib/.
                if hypre_dir.join("lib64").exists() {
                    lib_dirs.push(hypre_dir.join("lib64"));
                }
                if hypre_dir.join("lib").exists() {
                    lib_dirs.push(hypre_dir.join("lib"));
                }
                lib_dirs
            });
            let include_dir = include_dir.unwrap_or_else(|| hypre_dir.join("include"));
            (lib_dir, include_dir)
        }
    }
}

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    // println!("cargo:rustc-link-search=/path/to/lib");

    let (lib_dirs, include_dir) = find_hypre();

    if !lib_dirs.iter().all(|p| Path::new(p).exists()) {
        panic!("OpenSSL library directory does not exist: {:?}", lib_dirs);
    }
    if !Path::new(&include_dir).exists() {
        panic!(
            "OpenSSL include directory does not exist: {}",
            include_dir.to_string_lossy()
        );
    }

    for lib_dir in lib_dirs.iter() {
        println!(
            "cargo:rustc-link-search=native={}",
            lib_dir.to_string_lossy()
        );
    }
    println!("cargo:include={}", include_dir.to_string_lossy());

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=HYPRE");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let mut builder = bindgen::builder()
        // There is no need to make bindings for mpi types as that has already been done in the mpi crate
        .blocklist_item("(O?MPI|o?mpi)[\\w_]*")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate_comments(false);

    for inc_dir in &[include_dir] {
        builder = builder
            .clang_arg("-I")
            .clang_arg(inc_dir.display().to_string());
    }

    let bindings = builder
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .clang_args(&[
            "-I/usr/lib/x86_64-linux-gnu/openmpi/include",
            "-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi",
        ])
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
