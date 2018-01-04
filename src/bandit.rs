
use std::ops::{Index};

pub trait MultiArmedBandit<A: Clone + Index<A>> {
    fn select_arm(&self) -> A;
    fn update(&self, arm: A, reward: f32);
}
