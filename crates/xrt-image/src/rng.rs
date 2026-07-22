//! Versioned deterministic normal-noise generation.
//!
//! Version 1 uses SplitMix64 as an explicitly specified counter stream and a
//! Marsaglia polar transform implemented with the pinned pure-Rust `libm`
//! functions. It does not inherit behavior from `rand`'s default RNG.

pub const IMAGE_RNG_SCHEMA_V1: &str = "xrt-normal-v1-splitmix64-marsaglia-f32le";

#[derive(Debug, Clone)]
pub struct NormalRngV1 {
    seed: u64,
    counter: u64,
    spare: Option<f64>,
}

impl NormalRngV1 {
    pub const fn new(seed: u64) -> Self {
        Self {
            seed,
            counter: 0,
            spare: None,
        }
    }

    pub const fn counter(&self) -> u64 {
        self.counter
    }

    pub fn next_f32(&mut self) -> f32 {
        if let Some(spare) = self.spare.take() {
            return spare as f32;
        }
        loop {
            let u = 2.0 * self.uniform_open() - 1.0;
            let v = 2.0 * self.uniform_open() - 1.0;
            let radius = u * u + v * v;
            if radius > 0.0 && radius < 1.0 {
                let scale = libm::sqrt(-2.0 * libm::log(radius) / radius);
                self.spare = Some(v * scale);
                return (u * scale) as f32;
            }
        }
    }

    pub fn fill_f32(&mut self, output: &mut [f32]) {
        for value in output {
            *value = self.next_f32();
        }
    }

    pub fn fill_f32_le_bytes(&mut self, count: usize) -> Vec<u8> {
        let mut output = Vec::with_capacity(count.saturating_mul(4));
        for _ in 0..count {
            output.extend_from_slice(&self.next_f32().to_le_bytes());
        }
        output
    }

    fn uniform_open(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    fn next_u64(&mut self) -> u64 {
        let value = self
            .seed
            .wrapping_add(self.counter.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        self.counter = self.counter.wrapping_add(1);
        splitmix64(value)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_is_repeatable_and_little_endian() {
        let first = NormalRngV1::new(42).fill_f32_le_bytes(8);
        let golden = [
            0x07, 0x64, 0xfc, 0x3e, 0x5b, 0xaa, 0x31, 0xbf, 0x58, 0xfa, 0xa3, 0xbf, 0x74, 0xcc,
            0x66, 0xbf, 0xad, 0x14, 0x1a, 0xbf, 0x50, 0x7f, 0x24, 0x3f, 0x13, 0x6c, 0xc5, 0xbf,
            0x74, 0x21, 0x92, 0x3f,
        ];
        assert_eq!(first, golden);
        let second = NormalRngV1::new(42).fill_f32_le_bytes(8);
        assert_eq!(first, second);
        assert_ne!(first, NormalRngV1::new(43).fill_f32_le_bytes(8));
    }
}
