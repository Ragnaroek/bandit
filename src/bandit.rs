
pub trait MultiArmedBandit<A: Clone> {
    fn select_arm(&self) -> A;
    fn update(&mut self, arm: A, reward: f64);
}
