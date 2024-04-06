use crate::Tensor;

#[derive(Default, Clone)]
pub enum Init {
    Uniform,
    #[default]
    Xavier,
    Kaiming,
}

#[derive(Clone)]
pub enum Layer {
    Linear {
        input: usize,
        output: usize,
        weights: Tensor,
        biases: Tensor,
        gradient: Tensor,
    },
    Relu,
    Lrelu {
        a: f32,
    },
    Sigmoid,
    Tanh,
    Step,
    Softplus,
    Gelu,
    Softmax,
    Atan,
}

impl Layer {
    pub fn lin(input: usize, output: usize) -> Self {
        Self::Linear {
            input,
            output,
            weights: Tensor::default(),
            biases: Tensor::default(),
            gradient: Tensor::default(),
        }
    }

    pub fn init(&mut self, method: &Init) {
        match method {
            Init::Uniform => {
                *self.get_weights() =
                    Tensor::rand(&[self.get_input(), self.get_output()]) * 2.0 - 1.0;
            }
            Init::Xavier => {
                let range = (6f32 / (self.get_input() + self.get_output()) as f32).sqrt();
                *self.get_weights() =
                    (Tensor::rand(&[self.get_input(), self.get_output()]) * 2.0 - 1.0) * range;
            }
            Init::Kaiming => {
                let range = 2.0 / self.get_output() as f32;
                *self.get_weights() = Tensor::randn(&[self.get_input(), self.get_output()]) * range;
            }
        }
        *self.get_biases() = Tensor::zeros(&[self.get_output()]);
    }

    pub fn lin_in(input: usize) -> Self {
        Self::lin(input, 0)
    }

    pub fn lin_out(output: usize) -> Self {
        Self::lin(0, output)
    }

    pub fn get_output(&self) -> usize {
        match self {
            Layer::Linear {
                input: _, output, ..
            } => *output,
            _ => 0,
        }
    }

    pub fn set_output(&mut self, value: usize) {
        if let Layer::Linear { input, .. } = *self {
            *self = Self::lin(input, value);
        }
    }

    pub fn get_input(&self) -> usize {
        if let Layer::Linear { input, .. } = self {
            *input
        } else {
            0
        }
    }

    pub fn set_input(&mut self, value: usize) {
        if let Layer::Linear {
            input: _, output, ..
        } = self
        {
            *self = Self::lin(value, *output);
        }
    }

    pub fn has_weights(&self) -> bool {
        matches!(self, Layer::Linear { .. })
    }

    pub fn get_weights(&mut self) -> &mut Tensor {
        if let Layer::Linear { weights, .. } = self {
            weights
        } else {
            panic!("layer has no weights");
        }
    }

    pub fn get_biases(&mut self) -> &mut Tensor {
        if let Layer::Linear {
            weights: _, biases, ..
        } = self
        {
            biases
        } else {
            panic!("layer has no biases");
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Layer::Linear {
                input: _,
                output: _,
                weights,
                biases,
                ..
            } => &input.matmul(weights).squeeze_idx(0) + biases,
            Layer::Relu => input.apply(|x| x.max(0.0)),
            Layer::Lrelu { a } => input.apply(|x| if x >= 0.0 { x } else { x * a }),
            Layer::Sigmoid => input.apply(|x| 1.0 / (1.0 + (-x).exp())),
            Layer::Tanh => input.apply(|x| x.tanh()),
            Layer::Step => input.apply(|x| x.signum()),
            Layer::Softplus => input.apply(|x| (1.0 + x.exp()).ln()),
            Layer::Gelu => input.apply(|x| 0.5 * x * (1.0 + ((0.03 * x * x + 0.8) * x).tanh())),
            Layer::Softmax => input.softmax(),
            Layer::Atan => input.apply(|x| x.atan()),
        }
    }

    pub fn backward(&self, grad: &Tensor, out: &Tensor) -> Tensor {
        match self {
            Layer::Linear {
                input: _,
                output: _,
                weights,
                ..
            } => grad.dot(&weights),
            Layer::Relu => grad * &out.apply(|x| if x >= 0.0 { 1.0 } else { 0.0 }),
            Layer::Lrelu { a } => grad * &out.apply(|x| if x >= 0.0 { 1.0 } else { *a }),
            Layer::Sigmoid => {
                grad * &out.apply(|x| {
                    let s = 1.0 / (1.0 + (-x).exp());
                    s * (1.0 - s)
                })
            }
            Layer::Tanh => {
                grad * &out.apply(|x| {
                    let t = x.tanh();
                    1.0 - t * t
                })
            }
            Layer::Step => grad.clone(),
            Layer::Softplus => grad * &out.apply(|x| 1.0 / (1.0 + (-x).exp())),
            Layer::Gelu => {
                grad * &out.apply(|x| {
                    let a = x * (0.03 * x * x + 0.8);
                    let cosh_a = a.cosh();
                    0.5 * a.tanh() + x * (0.045 * x * x + 0.4) / (cosh_a * cosh_a) + 0.5
                })
            }
            Layer::Softmax => {
                let softmax_out = out.softmax();
                let mut result = Tensor::zeros(out.shape());
                for i in 0..out.numel() {
                    let mut sum = 0.0;
                    for j in 0..out.numel() {
                        if i == j {
                            sum += grad[j] * softmax_out[i] * (1.0 - softmax_out[i]);
                        } else {
                            sum -= grad[j] * softmax_out[i] * softmax_out[j];
                        }
                    }
                    result[i] = sum;
                }
                result
            }
            Layer::Atan => grad * &out.apply(|x| 1.0 / (x * x + 1.0)),
        }
    }

    pub fn update(&mut self, delta_weights: &Tensor, delta_biases: &Tensor) {
        if let Layer::Linear {
            input: _,
            output: _,
            weights,
            biases,
            ..
        } = self
        {
            *weights += delta_weights;
            *biases += delta_biases;
        }
    }
}
