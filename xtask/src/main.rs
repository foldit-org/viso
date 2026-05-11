use std::path::Path;
use std::process::Command;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build viso + viso-ui for WASM and assemble web/ directory for serving
    BuildWeb,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::BuildWeb => build_web(),
    }
}

/// WASM rustflags for multithreaded viso (SharedArrayBuffer + web workers).
const WASM_RUSTFLAGS: &str = "\
    -C target-feature=+atomics,+bulk-memory,+mutable-globals -C \
     link-arg=--shared-memory -C link-arg=--import-memory -C \
     link-arg=--max-memory=1073741824 -C link-arg=--export=__wasm_init_tls -C \
     link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C \
     link-arg=--export=__tls_base";

fn build_web() -> Result<()> {
    // Resolve the viso workspace root (parent of xtask/)
    let viso_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be inside the viso workspace");
    let web_dir = viso_root.join("web");

    // 1. Build viso cdylib for wasm32 (nightly, with atomics + shared memory)
    println!("Building viso for wasm32...");
    let status = Command::new("cargo")
        .args([
            "+nightly",
            "build",
            "--target",
            "wasm32-unknown-unknown",
            "--features",
            "web",
            "--no-default-features",
            "--release",
            "-Z",
            "build-std=panic_abort,std,alloc",
        ])
        .env("RUSTFLAGS", WASM_RUSTFLAGS)
        .current_dir(viso_root)
        .status()?;
    if !status.success() {
        anyhow::bail!("Failed to build viso for wasm32");
    }

    let wasm_path =
        viso_root.join("target/wasm32-unknown-unknown/release/viso.wasm");
    if !wasm_path.exists() {
        anyhow::bail!("WASM binary not found at {}", wasm_path.display());
    }

    // 2. Run wasm-bindgen to generate JS glue
    println!("Running wasm-bindgen...");
    let pkg_dir = web_dir.join("pkg");
    std::fs::create_dir_all(&pkg_dir)?;
    let status = Command::new("wasm-bindgen")
        .args(["--target", "web", "--out-dir"])
        .arg(&pkg_dir)
        .arg(&wasm_path)
        .status()?;
    if !status.success() {
        anyhow::bail!("wasm-bindgen failed");
    }

    // 3. Build viso-ui via trunk (single-threaded dioxus app, no special flags)
    let viso_ui_dir = viso_root.join("crates/viso-ui");
    println!("Building viso-ui...");
    let status = Command::new("trunk")
        .args(["build", "--release"])
        .current_dir(&viso_ui_dir)
        .status()?;
    if !status.success() {
        anyhow::bail!("trunk build failed for viso-ui");
    }

    // 4. Copy viso-ui dist to web/ui/
    let ui_dist = viso_ui_dir.join("dist");
    let ui_dst = web_dir.join("ui");
    if ui_dst.exists() {
        std::fs::remove_dir_all(&ui_dst)?;
    }
    copy_dir(&ui_dist, &ui_dst)?;

    println!("Web build complete.");
    println!(
        "  Serve with: cd {} && python3 -m http.server 8080",
        web_dir.display()
    );
    Ok(())
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let status = Command::new("cp").arg("-r").arg(src).arg(dst).status()?;
        if !status.success() {
            anyhow::bail!(
                "Failed to copy {} to {}",
                src.display(),
                dst.display()
            );
        }
    }
    #[cfg(windows)]
    {
        let status = Command::new("robocopy")
            .args([
                src.to_str().unwrap(),
                dst.to_str().unwrap(),
                "/E",
                "/NFL",
                "/NDL",
                "/NJH",
                "/NJS",
                "/NP",
            ])
            .status()?;
        match status.code() {
            Some(code) if code < 8 => {}
            _ => anyhow::bail!(
                "Failed to copy {} to {}",
                src.display(),
                dst.display()
            ),
        }
    }
    Ok(())
}
