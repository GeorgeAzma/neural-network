#[derive(Clone)]
pub struct Random {
    seed: u32,
}

impl Default for Random {
    fn default() -> Self {
        Self {
            seed: crate::seed::get(),
        }
    }
}

impl Random {
    pub fn next(&mut self) -> u32 {
        self.seed = self.seed.wrapping_add(0x9E3779B9);
        let mut x = self.seed;
        x ^= x >> 16;
        x = x.wrapping_mul(0x21F0AAAD);
        x ^= x >> 15;
        x = x.wrapping_mul(0x735A2D97);
        x ^= x >> 15;
        x
    }

    pub fn gen(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32
    }

    pub fn norm(&mut self) -> f32 {
        (-2.0 * self.gen().ln()).sqrt() * (std::f32::consts::TAU * self.gen()).cos()
    }

    pub fn gen_range(&mut self, range: std::ops::Range<f32>) -> f32 {
        self.gen() * (range.end - range.start) + range.start
    }

    pub fn gen_range_u32(&mut self, range: std::ops::Range<u32>) -> u32 {
        self.next() % (range.end - range.start) + range.start
    }

    pub fn gen_range_i32(&mut self, range: std::ops::Range<i32>) -> i32 {
        (self.next() % (range.end - range.start) as u32) as i32 + range.start
    }
}

pub trait Shuffle {
    fn shuffle(&mut self, r: &mut Random);
}

impl<T> Shuffle for &mut [T] {
    fn shuffle(&mut self, rng: &mut Random) {
        for i in 0..self.len() {
            let j = rng.gen_range_u32(i as u32..self.len() as u32);
            self.swap(i, j as usize);
        }
    }
}

impl<T> Shuffle for Vec<T> {
    fn shuffle(&mut self, rng: &mut Random) {
        (self as &mut [T]).shuffle(rng);
    }
}

pub fn new() -> Random {
    Default::default()
}
