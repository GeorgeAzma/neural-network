use crate::Tensor;
use crate::TensorRef;

pub trait Layer {
    fn forward(&self, input: &TensorRef) -> Tensor;

    fn backward(&self, _gradient: &TensorRef, _output: &TensorRef) -> Tensor {
        Tensor::default()
    }

    fn get_output(&self) -> usize {
        0
    }

    fn set_output(&mut self, _output: usize) {}

    fn get_input(&self) -> usize {
        0
    }

    fn set_input(&mut self, _input: usize) {}

    fn has_weights(&self) -> bool {
        false
    }

    fn update(&mut self, _delta_weights: &TensorRef, _delta_biases: &TensorRef) {}
}

pub struct Linear {
    input: usize,
    output: usize,
    weights: Tensor,
    biases: Tensor,
    gradient: Tensor,
}

impl Linear {
    pub fn new(input: usize, output: usize) -> Self {
        Self {
            input,
            output,
            weights: &Tensor::rand(&[input, output]) * 0.1,
            biases: Tensor::zeros(&[output]),
            gradient: Tensor::default(),
        }
    }

    pub fn input(input: usize) -> Self {
        Self::new(input, 0)
    }

    pub fn output(output: usize) -> Self {
        Self::new(0, output)
    }

    pub fn init(&mut self, mut f: impl FnMut(usize) -> f32) {
        self.weights
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = f(i));
    }
}

impl Layer for Linear {
    fn forward(&self, input: &TensorRef) -> Tensor {
        &input.matmul(&self.weights).squeeze() + &self.biases
    }

    fn backward(&self, gradient: &TensorRef, _output: &TensorRef) -> Tensor {
        gradient.matmul(&self.weights.t()).squeeze()
    }

    fn get_output(&self) -> usize {
        self.output
    }

    fn set_output(&mut self, output: usize) {
        self.output = output;
        *self = Self::new(self.input, output);
    }

    fn get_input(&self) -> usize {
        self.input
    }

    fn set_input(&mut self, input: usize) {
        self.input = input;
        *self = Self::new(input, self.output);
    }

    fn has_weights(&self) -> bool {
        true
    }

    fn update(&mut self, delta_weights: &TensorRef, delta_biases: &TensorRef) {
        self.weights += delta_weights;
        self.biases += delta_biases;
    }
}

pub struct Relu {}

impl Layer for Relu {
    fn forward(&self, inputs: &TensorRef) -> Tensor {
        inputs.apply(|x| if x >= 0.0 { x } else { 0.0 })
    }

    fn backward(&self, gradient: &TensorRef, output: &TensorRef) -> Tensor {
        gradient * &output.apply(|x| if x >= 0.0 { 1.0 } else { 0.0 })
    }
}

pub struct Lrelu {
    pub a: f32,
}

impl Lrelu {
    pub fn new(alpha: f32) -> Self {
        Self { a: alpha }
    }
}

impl Layer for Lrelu {
    fn forward(&self, inputs: &TensorRef) -> Tensor {
        inputs.apply(|x| if x >= 0.0 { x } else { x * self.a })
    }

    fn backward(&self, gradient: &TensorRef, output: &TensorRef) -> Tensor {
        gradient * &output.apply(|x| if x >= 0.0 { 1.0 } else { self.a })
    }
}

pub struct Sigmoid {}

impl Layer for Sigmoid {
    fn forward(&self, inputs: &TensorRef) -> Tensor {
        inputs.apply(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn backward(&self, gradient: &TensorRef, output: &TensorRef) -> Tensor {
        gradient
            * &output.apply(|x| {
                let x = 1.0 / (1.0 + (-x).exp());
                x * (1.0 - x)
            })
    }
}

pub struct Tanh {}

impl Layer for Tanh {
    fn forward(&self, inputs: &TensorRef) -> Tensor {
        inputs.apply(|x| x.tanh())
    }

    fn backward(&self, gradient: &TensorRef, output: &TensorRef) -> Tensor {
        gradient * &output.apply(|x| 1.0 - x.tanh() * x.tanh())
    }
}

pub struct Step {}

impl Layer for Step {
    fn forward(&self, inputs: &TensorRef) -> Tensor {
        inputs.apply(|x| x.signum())
    }

    fn backward(&self, gradient: &TensorRef, output: &TensorRef) -> Tensor {
        gradient * &output.apply(|_| 1.0)
    }
}

pub struct Softplus {}

impl Layer for Softplus {
    fn forward(&self, inputs: &TensorRef) -> Tensor {
        inputs.apply(|x| (1.0 + x.exp()).ln())
    }

    fn backward(&self, gradient: &TensorRef, output: &TensorRef) -> Tensor {
        gradient * &output.apply(|x| 1.0 / (1.0 + (-x).exp()))
    }
}

pub struct Gelu {}

impl Layer for Gelu {
    fn forward(&self, inputs: &TensorRef) -> Tensor {
        inputs.apply(|x| 0.5 * x * (1.0 + ((0.8 + 0.03 * x * x) * x).tanh()))
    }

    fn backward(&self, gradient: &TensorRef, output: &TensorRef) -> Tensor {
        gradient
            * &output.apply(|x| {
                let a = x * (0.035 * x * x + 0.8);
                let cosh_a = a.cosh();
                0.5 * a.tanh() + 0.5 + x * (0.05 * x * x + 0.4) / (cosh_a * cosh_a)
            })
    }
}

pub struct Softmax {}

impl Layer for Softmax {
    fn forward(&self, input: &TensorRef) -> Tensor {
        input.softmax()
    }

    fn backward(&self, gradient: &TensorRef, _output: &TensorRef) -> Tensor {
        // TODO: using actual softmax derivative had bad results
        gradient.to_owned()
    }
}

pub struct Atan {}

impl Layer for Atan {
    fn forward(&self, inputs: &TensorRef) -> Tensor {
        inputs.apply(|x| x.atan())
    }

    fn backward(&self, gradient: &TensorRef, output: &TensorRef) -> Tensor {
        gradient * &output.apply(|x| 1.0 / (x.atan().powi(2) + 1.0))
    }
}
