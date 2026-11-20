# hypre-rs: A Rust Interface to the hypre Library
This crate provides a Rust interface to the [hypre](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods) library.

## Documentation
The API documentation of the `master` branch is published at
<https://cedricchevalier19.github.io/hypre-rs/>. It is rebuilt on every push to `master`.

To build it locally:
```bash
cargo doc --no-deps --all-features -p hypre-rs --open
```

## Getting Started
### Prerequisites
- The hypre library must be installed on your system before you can use hypre-rs.
- You can install hypre using your system's package manager or build it from source.
- On Ubuntu or Debian, install the hypre library and development package with:

```bash
sudo apt install libhypre-dev
```

### Installation
Since hypre-rs is not yet released, you will need to install it from GitHub.

Follow these steps to install the crate:
1. Clone the GitHub repository:
```bash
 git clone https://github.com/cedricchevalier19/hypre-rs.git
```
2. Change into the new directory:
```bash
cd hypre-rs
```
3. Build the project in release mode:
```bash
cargo build --release
```

If your hypre installation is not in a standard location, you must set the `HYPRE_DIR` environment variable to the path of your hypre installation. Here is an example with hypre installed in `/opt/hypre`:
```bash
export HYPRE_DIR=/opt/hypre
```

For a programmatic approach, set the `HYPRE_DIR` environment variable in your `~/.cargo/config` or project's `.cargo/config.toml' file. Here is how you add it to the `~/.cargo/config` file if hypre is installed in `/opt/hypre`:
```toml
[env]
HYPRE_DIR = "/opt/hypre"
```


### Development container
A ready-to-use development environment (Rust, hypre, OpenMPI, libclang) is provided in `.devcontainer/`.
It works on both x86_64 and arm64 hosts, including Apple Silicon.

- **VS Code / JetBrains IDEs**: open the project and choose *Reopen in Container*.
- **Plain Docker** (`CARGO_TARGET_DIR` keeps the Linux build output out of your host's `target/`):
  ```bash
  docker build -t hypre-rs-dev .devcontainer
  docker run --rm -it -e CARGO_TARGET_DIR=/home/dev/target -v "$PWD":/workspaces/hypre-rs hypre-rs-dev
  # then, inside the container:
  cargo test
  cargo run --example simple-cg
  mpirun -n 4 /home/dev/target/debug/examples/simple-cg
  ```
- **Apple `container`** (macOS 26+): same commands, with `container` in place of `docker`
  (run `container system start` once first).

The image sets `HYPRE_LIB_DIR` and `HYPRE_INCLUDE_DIR`, which take precedence over any `HYPRE_DIR` in `.cargo/config.toml`.

## Examples

The [`examples`](examples) directory has complete programs. Each runs on any
number of MPI ranks and checks its own result:

| Example | Shows |
|---|---|
| [`simple-cg`](examples/simple-cg.rs) | The minimal workflow: a symmetric matrix, CG, and its status |
| [`pcg-preconditioners`](examples/pcg-preconditioners.rs) | CG on a 2D diffusion problem with a coefficient jump: no preconditioner, Jacobi, algebraic multigrid |
| [`gmres-convection-diffusion`](examples/gmres-convection-diffusion.rs) | A non-symmetric system with GMRES and BiCGSTAB, preconditioned by Jacobi or multigrid |
| [`amg-solver`](examples/amg-solver.rs) | Algebraic multigrid as a solver: named options, reading the configuration back, handling non-convergence and restarting |
| [`choose-solver`](examples/choose-solver.rs) | Choosing the solver at runtime, from the command line |
| [`check-symmetry`](examples/check-symmetry.rs) | Checking that a matrix built with both triangles is symmetric, to use CG |

```bash
cargo run --example pcg-preconditioners
cargo build --examples && mpirun -n 4 target/debug/examples/pcg-preconditioners
```

## License
This project is licensed under dual MIT/Apache-2.0 license. See the [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) files for details.

The SPDX license identifier for the license chosen is: `MIT OR Apache-2.0`.
