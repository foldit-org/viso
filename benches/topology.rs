//! Viso topology re-derivation bench: how fast viso rebuilds per-entity
//! render topology from a molex `Assembly` — the work
//! `SyncPipeline::sync_from_assembly` runs on every load and every
//! topology-changing mutation. This is the viso-side half of the
//! "file -> renderable" budget; the molex parse benches cover the other
//! half. (Full budget = parse + `recompute_ss` + this.)
//!
//! Two measurements per fixture, to separate the SoA-relevant cost from
//! the rest:
//! - `derive_topology`: the full per-entity derivation (element transpose,
//!   backbone/sidechain layouts, residue tables, bond/SS clone).
//! - `extract_positions`: just the per-entity `positions()` gather — the pure
//!   AoS->SoA transpose baseline.
//!
//! Fixtures reuse the molex bench corpus (`crates/molex/benches/data`,
//! 1ubq/4hhb committed; 6vxx via `MOLEX_BENCH_LARGE_DIR`), parsed from
//! mmCIF and given secondary structure via `recompute_ss` so the SS clone
//! is load-realistic.

#![allow(unused_results, missing_docs)]

use std::path::{Path, PathBuf};

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use molex::Assembly;

struct Fixture {
    name: &'static str,
    assembly: Assembly,
}

/// The molex bench fixtures live next door (viso is a sibling crate).
fn molex_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../molex/benches/data")
}

#[expect(
    clippy::panic,
    reason = "bench setup; a missing/invalid fixture is a hard error"
)]
fn assembly_from_cif(path: &Path) -> Assembly {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let entities = molex::adapters::cif::mmcif_str_to_entities(&text)
        .unwrap_or_else(|e| panic!("parse {}: {e:?}", path.display()));
    let mut assembly = Assembly::new(entities);
    // Load-realistic: the host runs recompute_ss before publishing to viso.
    assembly.recompute_ss();
    assembly
}

fn load_fixtures() -> Vec<Fixture> {
    let dir = molex_data_dir();
    let mut out = Vec::new();
    for name in ["1ubq", "4hhb"] {
        out.push(Fixture {
            name,
            assembly: assembly_from_cif(&dir.join(format!("{name}.cif"))),
        });
    }
    if let Some(large) = std::env::var_os("MOLEX_BENCH_LARGE_DIR") {
        let path = PathBuf::from(large).join("6vxx.cif");
        if path.exists() {
            out.push(Fixture {
                name: "6vxx",
                assembly: assembly_from_cif(&path),
            });
        }
    }
    out
}

fn bench_topology(c: &mut Criterion) {
    let fixtures = load_fixtures();
    let mut group = c.benchmark_group("viso_topology");
    for fx in &fixtures {
        group.bench_with_input(
            BenchmarkId::new("derive_topology", fx.name),
            &fx.assembly,
            |b, asm| {
                b.iter(|| {
                    black_box(viso::bench_api::derive_topology_all(asm));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("extract_positions", fx.name),
            &fx.assembly,
            |b, asm| {
                b.iter(|| {
                    for entity in asm.entities() {
                        black_box(entity.positions());
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_topology);
criterion_main!(benches);
