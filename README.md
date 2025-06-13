# qsv-stats

This library provides common statistical functions with support for computing them
efficiently on *streams* of data. The intent is to permit parallel computation of
statistics on large data sets.

> NOTE: This fork of [streaming-stats](https://github.com/BurntSushi/rust-stats) merges 
pending upstream PRs for quartile computation and a different variance algorithm 
that is used in [qsv](https://github.com/dathere/qsv)'s [`stats` command](https://github.com/dathere/qsv/blob/master/README.md#stats_deeplink).  
It has numerous other stats, heavily updated for performance, uses parallel processing,
uses the [fused multiply add](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation#Fused_multiply%E2%80%93add) CPU instruction along with several other performance tweaks.

Dual-licensed under MIT or the [UNLICENSE](http://unlicense.org).


### Documentation

Documentation for qsv-stats exists here:
[https://docs.rs/qsv-stats](https://docs.rs/qsv-stats).


### Installation

Simply add `qsv-stats` as a dependency to your project's `Cargo.toml`:

```toml
[dependencies]
qsv-stats = "0.33"
```
