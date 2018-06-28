[![Build Status](https://travis-ci.org/Ragnaroek/bandit.svg?branch=master)](https://travis-ci.org/Ragnaroek/bandit)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/Ragnaroek/bandit/blob/master/LICENSE)
[![](http://meritbadge.herokuapp.com/bandit)](https://crates.io/crates/bandit)

# Multi-armed bandit algorithms in Rust

## Cargo

```toml
[dependencies]
bandit = "0.12.0"
```

## Description and Scope

This library currently only implements the annealing softmax bandit algorithm.
Future work may also implement other bandit algorithm variants (pull-requests are welcomed).

## Usage and Configuration

First, you need to create a bandit with three parameters:

```rust
let bandit = AnnealingSoftmax::new(arms, DEFAULT_BANDIT_CONFIG.clone(), DEFAULT_CONFIG);
```

The first parameters is the list of arms the bandit will draw from. An arm can be anything
that implements the `Clone + Hash + Eq + Identifiable` traits. You probably always will
derive the first three traits, but the last one, `Identifiable`, is special.

```rust
pub trait Identifiable {
    fn ident(&self) -> String;
}
```
Well not very special, it should be very easy to implement. The reason for this additional trait is
that the full bandit state can be persisted to disk (in JSON format), to later continue at the
exact state of the algorithm. Unfortunately ```Hash``` makes no guarantee about the used hashing
algorithm and may change between Rust versions. Since we want to be able to load states regardless of the
Rust version, we require the trait returning a unique and easily serialisable `String` as a unique identifier
for the arm.

The next parameter is the general bandit configuration:
```rust
#[derive(Debug, PartialEq, Clone)]
pub struct BanditConfig {
    /// Log file for logging details about the bandit algorithm
    /// run. What will be logged depends on the bandit algorithm
    /// implementation.
    pub log_file: Option<PathBuf>
}
```

Allowing you to optionally supply a path to a log file, where every step in the algorithm is logged to.
The log file is a simple csv file, logging arm draw states and updates to the rewards:
```csv
SELECT;threads:5;1529824600935
SELECT;threads:18;1529825031483
UPDATE;threads:18;1529825931520;0.1320011111111111
SELECT;threads:9;1529825931560
UPDATE;threads:9;1529826831455;0.1326688888888889
SELECT;threads:14;1529826831506
UPDATE;threads:14;1529827731423;0.13315000000000002
SELECT;threads:7;1529827731455
UPDATE;threads:7;1529828631384;0.12791666666666668
...
```

The last parameter is a special configuration for the annealing softmax algorithm:
```rust
#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct AnnealingSoftmaxConfig {
    /// The higher the value the faster the algorithms tends toward selecting
    /// the arm with highest reward. Should be a number between [0, 1.0)
    pub cooldown_factor : f64
}
```

It currently only has one option: the `cooldown_factor`, which may be a float between
0 and 1.0. You can control how fast the annealing will happen with this factor.
The higher the `cooldown_factor` the faster the algorithm will stop exploring new arms and will
stick with the arm with the highest reward discovered so far. You probably have to experiment
with this factor to find the best one for your particular setup (there are also tools to help you with that, see below).

After constructing and configuring your bandit, you can start selecting arms:
```rust
let arm = bandit.select_arm();
```

and update the reward for arms:
```rust
let reward = ... some f64 value
bandit.update(arm, reward);
```

Thats basically it. At some point and after enough rewards (and depending on your chosen `cooldown_factor`)
the system will be completely cooled off and always selecting the highest reward arm. It is safe to
let the system run for a very long time, always selecting arms and updating without fears for overflow
errors and inconsistent bandit states (this was found out the hard way in a unit test, hooray for unit-testing).

## Saving and Restoring states

A bandit can save itself into a file:

```rust
bandit.save_bandit(<path to save file>);
```

The bandit can be loaded again from the particular implementation:

```rust
let arms = ... list of arms, as in the initial creation
let bandit_config = ... BanditConfig, like in second parameter from initial creation
let bandit_loaded = AnnealingSoftmax::load_bandit(arms, bandit_config, <path to save file>);
```

The arms supplied do not necessarily have to match the arms that are restored from the file.
If an arm is removed, it will be removed after loading. You will loose the stored reward after
saving the bandit again. If a new arm is added, it will start with a zero reward.

## Visualising Bandit Arm Selection Data

TODO Include teaser picture here.

The bandit tools application allows you to analyse the stored state file and log file.
Details are described in the separate repo: TODO Link to bandit_tool repo
