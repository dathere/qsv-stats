An experimental library that provides some common statistical functions with
some support for computing them efficiently on *streams* of data. The intent
is to permit parallel computation of statistics on large data sets.

> NOTE: This fork merges pending upstream PRs for quartile computation and a different variance algorithm that is used in [qsv](https://github.com/jqnatividad/qsv) stats.

Dual-licensed under MIT or the [UNLICENSE](http://unlicense.org).


### Documentation

Some documentation exists here:
[https://docs.rs/streaming-stats](https://docs.rs/streaming-stats).


### Installation

Simply add `streaming-stats` as a dependency to your project's `Cargo.toml`:

```toml
[dependencies]
streaming-stats = "0.2"
```
