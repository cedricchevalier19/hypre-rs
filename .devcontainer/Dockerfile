# Development image for hypre-rs.
#
# Provides: stable Rust (+ clippy, rustfmt), hypre, OpenMPI, and libclang for bindgen.
# Works on both amd64 and arm64 (e.g. Apple Silicon) hosts.
ARG RUST_VERSION=1
FROM rust:${RUST_VERSION}-bookworm

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libhypre-dev \
      openmpi-bin libopenmpi-dev \
      clang libclang-dev \
      pkg-config \
      gdb lldb \
      sudo less vim-tiny \
 && rm -rf /var/lib/apt/lists/*

# Debian installs hypre and OpenMPI under an architecture-specific directory
# (/usr/lib/x86_64-linux-gnu, /usr/lib/aarch64-linux-gnu, ...). Expose them at
# fixed paths so the environment variables below work on any architecture.
RUN set -eux; \
    triple="$(gcc -print-multiarch)"; \
    mkdir -p /opt/hypre; \
    ln -s /usr/include/hypre /opt/hypre/include; \
    ln -s "/usr/lib/${triple}" /opt/hypre/lib; \
    ln -s "/usr/lib/${triple}/openmpi" /opt/openmpi; \
    test -f /opt/hypre/include/HYPRE.h; \
    test -f /opt/openmpi/include/mpi.h

# hypre-sys/build.rs reads HYPRE_LIB_DIR / HYPRE_INCLUDE_DIR first. These also
# take precedence over the HYPRE_DIR=/opt/homebrew set in .cargo/config.toml,
# because Cargo's [env] table never overrides variables already set.
# BINDGEN_EXTRA_CLANG_ARGS lets bindgen find mpi.h on non-x86_64 hosts, since
# build.rs only hard-codes the x86_64 OpenMPI include paths.
ENV HYPRE_LIB_DIR=/opt/hypre/lib \
    HYPRE_INCLUDE_DIR=/opt/hypre/include \
    BINDGEN_EXTRA_CLANG_ARGS="-I/opt/openmpi/include -I/opt/openmpi/include/openmpi" \
    # Let `mpirun -n N` use more ranks than there are cores, and avoid the
    # shared-memory (CMA) warnings OpenMPI prints inside containers.
    OMPI_MCA_rmaps_base_oversubscribe=1 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none

RUN rustup component add clippy rustfmt rust-src

# Non-root user (OpenMPI refuses to run as root by default).
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
 && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" \
 && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
 && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
 && chown -R "${USERNAME}:${USERNAME}" "${CARGO_HOME}" "${RUSTUP_HOME}"

USER ${USERNAME}
WORKDIR /workspaces/hypre-rs
