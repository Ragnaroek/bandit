
use super::bandit::{MultiArmedBandit};
use std::ops::{Index};

struct AnnealingSoftmax<A> {
    arms: [A]
}

impl<A: Clone + Index<A>> MultiArmedBandit<A> for AnnealingSoftmax<A> {

    fn select_arm(&self) -> A {
        return self.arms[0].clone();
    }

    fn update(&self, arm: A, reward: f32) {

    }
}
