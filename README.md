This library provides common statistical functions with support for computing them
efficiently on *streams* of data. The intent is to permit parallel computation of
statistics on large data sets.

> NOTE: This fork of [streaming-stats](https://github.com/BurntSushi/rust-stats) merges 
pending upstream PRs for quartile computation and a different variance algorithm 
that is used in [qsv](https://github.com/jqnatividad/qsv) stats.<br><br>
It has also been updated to Rust 2021 edition, uses the fused multiply add CPU instruction
along with some other performance tweaks.
This is being published on crates.io, so that qsv can be published as well.

Dual-licensed under MIT or the [UNLICENSE](http://unlicense.org).


### Documentation

Original documentation for streaming-stats exists here:
[https://docs.rs/streaming-stats](https://docs.rs/streaming-stats).


### Installation

Simply add `qsv-stats` as a dependency to your project's `Cargo.toml`:

```toml
[dependencies]
qsv-stats = "0.19"
```
