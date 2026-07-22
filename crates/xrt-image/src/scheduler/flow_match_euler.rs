use serde::{Deserialize, Serialize};

use crate::ImageError;

fn default_train_timesteps() -> usize {
    1_000
}

fn default_shift() -> f64 {
    1.0
}

fn default_base_seq_len() -> usize {
    256
}

fn default_max_seq_len() -> usize {
    4_096
}

fn default_base_shift() -> f64 {
    0.5
}

fn default_max_shift() -> f64 {
    1.15
}

fn default_time_shift_type() -> String {
    "exponential".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowMatchEulerConfig {
    #[serde(default = "default_train_timesteps")]
    pub num_train_timesteps: usize,
    #[serde(default = "default_shift")]
    pub shift: f64,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    #[serde(default = "default_base_seq_len")]
    pub base_image_seq_len: usize,
    #[serde(default = "default_max_seq_len")]
    pub max_image_seq_len: usize,
    #[serde(default = "default_base_shift")]
    pub base_shift: f64,
    #[serde(default = "default_max_shift")]
    pub max_shift: f64,
    #[serde(default)]
    pub shift_terminal: Option<f64>,
    #[serde(default = "default_time_shift_type")]
    pub time_shift_type: String,
    #[serde(default)]
    pub invert_sigmas: bool,
    #[serde(default)]
    pub use_karras_sigmas: bool,
    #[serde(default)]
    pub use_exponential_sigmas: bool,
    #[serde(default)]
    pub use_beta_sigmas: bool,
}

impl FlowMatchEulerConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ImageError> {
        let config: Self = serde_json::from_slice(bytes)
            .map_err(|error| ImageError::Manifest(format!("invalid scheduler config: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ImageError> {
        if self.num_train_timesteps == 0 {
            return Err(ImageError::Manifest(
                "scheduler num_train_timesteps must be positive".to_string(),
            ));
        }
        for (name, value) in [
            ("shift", self.shift),
            ("base_shift", self.base_shift),
            ("max_shift", self.max_shift),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ImageError::Manifest(format!(
                    "scheduler {name} must be finite and positive"
                )));
            }
        }
        if self.base_image_seq_len >= self.max_image_seq_len {
            return Err(ImageError::Manifest(
                "scheduler base_image_seq_len must be below max_image_seq_len".to_string(),
            ));
        }
        if self.use_dynamic_shifting
            && !matches!(self.time_shift_type.as_str(), "exponential" | "linear")
        {
            return Err(ImageError::Manifest(format!(
                "unsupported dynamic time_shift_type `{}`",
                self.time_shift_type
            )));
        }
        if self
            .shift_terminal
            .is_some_and(|value| !value.is_finite() || !(0.0..1.0).contains(&value))
        {
            return Err(ImageError::Manifest(
                "scheduler shift_terminal must be finite in [0, 1)".to_string(),
            ));
        }
        let alternate_count = usize::from(self.use_karras_sigmas)
            + usize::from(self.use_exponential_sigmas)
            + usize::from(self.use_beta_sigmas);
        if alternate_count > 1 {
            return Err(ImageError::Manifest(
                "only one alternate sigma schedule may be enabled".to_string(),
            ));
        }
        if alternate_count != 0 {
            return Err(ImageError::UnsupportedCapability(
                "Karras, exponential, and beta FlowMatch schedules are not admitted yet"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowMatchEulerSchedule {
    config: FlowMatchEulerConfig,
    image_seq_len: usize,
    mu: Option<f64>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

impl FlowMatchEulerSchedule {
    pub fn new(
        config: FlowMatchEulerConfig,
        steps: usize,
        image_seq_len: usize,
    ) -> Result<Self, ImageError> {
        config.validate()?;
        if steps == 0 || image_seq_len == 0 {
            return Err(ImageError::InvalidRequest(
                "scheduler steps and image sequence length must be positive".to_string(),
            ));
        }
        if steps == 1 && config.shift_terminal.is_some() {
            return Err(ImageError::InvalidRequest(
                "FlowMatch schedules with shift_terminal require at least two inference steps"
                    .to_string(),
            ));
        }
        let mu = config
            .use_dynamic_shifting
            .then(|| calculate_mu(&config, image_seq_len));
        let mut sigmas = (0..steps)
            .map(|index| {
                1.0 - index as f32 * (1.0 - 1.0 / steps as f32) / (steps - 1).max(1) as f32
            })
            .collect::<Vec<_>>();
        if steps == 1 {
            sigmas[0] = 1.0;
        }
        if let Some(mu) = mu {
            let shift = libm::exp(mu) as f32;
            for sigma in &mut sigmas {
                *sigma = match config.time_shift_type.as_str() {
                    "exponential" => shift / (shift + (1.0 / *sigma - 1.0)),
                    "linear" => {
                        let mu = mu as f32;
                        mu / (mu + (1.0 / *sigma - 1.0))
                    }
                    _ => unreachable!("validated time shift type"),
                };
            }
        } else if config.shift != 1.0 {
            let shift = config.shift as f32;
            for sigma in &mut sigmas {
                *sigma = shift * *sigma / (1.0 + (shift - 1.0) * *sigma);
            }
        }
        if let Some(terminal) = config.shift_terminal {
            let final_one_minus = 1.0 - *sigmas.last().expect("non-empty schedule");
            let scale = final_one_minus / (1.0 - terminal as f32);
            for sigma in &mut sigmas {
                *sigma = 1.0 - (1.0 - *sigma) / scale;
            }
        }
        if sigmas.iter().any(|sigma| !sigma.is_finite()) {
            return Err(ImageError::Numerical {
                component: "flow_match_euler_schedule",
                step: 0,
            });
        }
        let timesteps = sigmas
            .iter()
            .map(|sigma| *sigma * config.num_train_timesteps as f32)
            .collect::<Vec<_>>();
        sigmas.push(if config.invert_sigmas { 1.0 } else { 0.0 });
        Ok(Self {
            config,
            image_seq_len,
            mu,
            sigmas,
            timesteps,
        })
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub const fn image_seq_len(&self) -> usize {
        self.image_seq_len
    }

    pub const fn mu(&self) -> Option<f64> {
        self.mu
    }

    pub fn step(
        &self,
        step: usize,
        model_output: &[f32],
        sample: &mut [f32],
    ) -> Result<(), ImageError> {
        if step >= self.timesteps.len() || model_output.len() != sample.len() {
            return Err(ImageError::Internal(format!(
                "invalid scheduler step {step} or tensor length mismatch"
            )));
        }
        let delta = self.sigmas[step + 1] - self.sigmas[step];
        for (sample, prediction) in sample.iter_mut().zip(model_output) {
            *sample += delta * *prediction;
            if !sample.is_finite() {
                return Err(ImageError::Numerical {
                    component: "flow_match_euler",
                    step,
                });
            }
        }
        Ok(())
    }
}

fn calculate_mu(config: &FlowMatchEulerConfig, image_seq_len: usize) -> f64 {
    let slope = (config.max_shift - config.base_shift)
        / (config.max_image_seq_len - config.base_image_seq_len) as f64;
    let intercept = config.base_shift - slope * config.base_image_seq_len as f64;
    image_seq_len as f64 * slope + intercept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pinned_qwen_512_four_step_schedule() {
        let config = FlowMatchEulerConfig {
            num_train_timesteps: 1_000,
            shift: 1.0,
            use_dynamic_shifting: true,
            base_image_seq_len: 256,
            max_image_seq_len: 8_192,
            base_shift: 0.5,
            max_shift: 0.9,
            shift_terminal: Some(0.02),
            time_shift_type: "exponential".to_string(),
            invert_sigmas: false,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
        };
        let schedule = FlowMatchEulerSchedule::new(config, 4, 1_024).unwrap();
        let expected = [1.0, 0.749_268_23, 0.432_587_92, 0.019_999_921, 0.0];
        for (actual, expected) in schedule.sigmas().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert!((schedule.mu().unwrap() - 0.538_709_677_419_354_8).abs() < 1e-12);
    }

    #[test]
    fn euler_step_uses_adjacent_sigmas() {
        let config = FlowMatchEulerConfig {
            num_train_timesteps: 1_000,
            shift: 1.0,
            use_dynamic_shifting: false,
            base_image_seq_len: 256,
            max_image_seq_len: 4_096,
            base_shift: 0.5,
            max_shift: 1.15,
            shift_terminal: None,
            time_shift_type: "exponential".to_string(),
            invert_sigmas: false,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
        };
        let schedule = FlowMatchEulerSchedule::new(config, 2, 1).unwrap();
        let mut sample = [1.0, 2.0];
        schedule.step(0, &[2.0, -2.0], &mut sample).unwrap();
        assert_eq!(sample, [0.0, 3.0]);
    }

    #[test]
    fn terminal_shift_rejects_a_degenerate_one_step_schedule() {
        let config = FlowMatchEulerConfig {
            num_train_timesteps: 1_000,
            shift: 1.0,
            use_dynamic_shifting: true,
            base_image_seq_len: 256,
            max_image_seq_len: 8_192,
            base_shift: 0.5,
            max_shift: 0.9,
            shift_terminal: Some(0.02),
            time_shift_type: "exponential".to_string(),
            invert_sigmas: false,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
        };
        let error = FlowMatchEulerSchedule::new(config, 1, 1).unwrap_err();
        assert!(matches!(error, ImageError::InvalidRequest(_)));
    }
}
