# Suggested Commands

## Development
- `cargo build` - Debug build
- `cargo build --release` - Release build
- `cargo test --lib` - Run all library tests
- `cargo test <name>` - Run specific test
- `cargo clippy --lib` - Lint check
- `cargo fmt` - Format code
- `cargo doc --open` - Generate docs

## Benchmarking
- `cargo test comprehensive_quartiles_benchmark -- --ignored --nocapture` - Quartile benchmarks
- `hyperfine` for comparing branches (see .claude/skills/bench-compare)

## System (Darwin)
- `git`, `ls`, `grep`, `find` - standard unix tools