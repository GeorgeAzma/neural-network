use crate::layer::*;
use crate::Tensor;

#[derive(Clone, Default)]
pub struct Model {
    layers: Vec<Layer>,
    delta_weights: Vec<Tensor>,
    delta_biases: Vec<Tensor>,
    init_method: Init,
}

impl Model {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn init(&mut self, method: Init) {
        self.init_method = method;
    }

    pub fn add_layer(&mut self, mut layer: Layer) {
        for prev_layer in self.layers.iter_mut().rev() {
            if prev_layer.get_input() > 0 || prev_layer.get_output() > 0 {
                if prev_layer.get_output() > 0 {
                    assert!(
                        layer.get_input() == 0 || layer.get_input() == prev_layer.get_output(),
                        "layer input doesn't match previous layer output ({} != {})",
                        layer.get_input(),
                        prev_layer.get_output(),
                    );
                    layer.set_input(prev_layer.get_output());
                }
                if layer.get_input() > 0 {
                    assert!(
                        layer.get_output() == 0 || layer.get_input() == prev_layer.get_output(),
                        "layer input doesn't match previous layer output ({} != {})",
                        layer.get_input(),
                        prev_layer.get_output(),
                    );
                    prev_layer.set_output(layer.get_input());
                    prev_layer.init(&self.init_method);
                }
                break;
            }
        }
        self.delta_weights.push(Tensor::default());
        self.delta_biases.push(Tensor::default());
        if layer.get_input() > 0 && layer.get_output() > 0 {
            layer.init(&self.init_method);
        }
        self.layers.push(layer);
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        let mut inputs = inputs.to_owned();
        for layer in self.layers.iter() {
            inputs = layer.forward(&inputs);
        }
        inputs
    }

    pub fn step(&mut self, input: &Tensor, target: &Tensor) -> Tensor {
        let mut output = Vec::new();
        output.resize(self.layers.len(), Tensor::default());
        for (i, layer) in self.layers.iter().enumerate() {
            let input = if i == 0 { input } else { &output[i - 1] };
            output[i] = layer.forward(input);
        }

        let mut gradient = target - output.last().unwrap();
        for (i, layer) in self.layers.iter().enumerate().rev() {
            if layer.has_weights() {
                let input = if i == 0 { input } else { &output[i - 1] };
                if self.delta_weights[i].numel() == 0 {
                    self.delta_weights[i] = Tensor::zeros(&[input.numel(), gradient.numel()]);
                }
                if self.delta_biases[i].numel() == 0 {
                    self.delta_biases[i] = Tensor::zeros(&[gradient.numel()]);
                }
                self.delta_weights[i] += &input.outer(&gradient);
                self.delta_biases[i] += &gradient;
            }
            if i == 0 {
                let default = Tensor::default();
                gradient = layer.backward(&gradient, &default);
            } else {
                gradient = layer.backward(&gradient, &output[i - 1]);
            }
        }

        output.last().unwrap().to_owned()
    }

    pub fn update(&mut self, learning_rate: f32) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if layer.has_weights() {
                self.delta_weights[i] *= learning_rate;
                self.delta_biases[i] *= learning_rate;
                layer.update(&self.delta_weights[i], &self.delta_biases[i]);
                self.delta_weights[i].fill_(0.0);
                self.delta_biases[i].fill_(0.0);
            }
        }
    }
}
