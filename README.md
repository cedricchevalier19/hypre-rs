# hypre-rs: A Rust Interface to the hypre Library
This crate provides a Rust interface to the [hypre](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods) library.

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


## License
This project is licensed under dual MIT/Apache-2.0 license. See the [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) files for details.

The SPDX license identifier for the license chosen is: `MIT OR Apache-2.0`.
