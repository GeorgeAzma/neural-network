use crate::layer::*;
use crate::Slice;
use crate::Tensor;

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    delta_weights: Vec<Tensor>,
    delta_biases: Vec<Tensor>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            delta_weights: Vec::new(),
            delta_biases: Vec::new(),
        }
    }

    pub fn add_layer<L: Layer + 'static>(&mut self, mut layer: L) {
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
                }
                break;
            }
        }
        self.delta_weights.push(Tensor::default());
        self.delta_biases.push(Tensor::default());
        self.layers.push(Box::new(layer));
    }

    pub fn forward<T: Slice<f32>, S: Slice<usize>>(&self, inputs: &Tensor<T, S>) -> Tensor {
        let mut inputs = inputs.to_owned();
        for layer in self.layers.iter() {
            inputs = layer.forward(&inputs.as_ref());
        }
        inputs
    }

    pub fn step<T1: Slice<f32>, S1: Slice<usize>, T2: Slice<f32>, S2: Slice<usize>>(
        &mut self,
        inputs: &Tensor<T1, S1>,
        targets: &Tensor<T2, S2>,
    ) -> Tensor {
        let mut outputs = Vec::new();
        outputs.resize(self.layers.len(), Tensor::default());
        for (i, layer) in self.layers.iter().enumerate() {
            let input = if i > 0 {
                outputs[i - 1].as_ref()
            } else {
                inputs.as_ref()
            };
            outputs[i] = layer.forward(&input);
        }
        // outputs: [z1, a1, z2, a2]
        // gradients:
        // 4. z2_dt softmax.backward((y - o), a2)
        // 3. a1_dt l2.backward(z2_dt, z2) # has_weights
        // 2. z1_dt relu.backward(a1_dt, a1)
        // 1. in_dt l1.backward(z1_dt, z1) # has_weights
        let mut gradient = targets - outputs.last().unwrap();

        for (i, layer) in self.layers.iter().rev().enumerate() {
            let i = outputs.len() - 1 - i;
            if layer.has_weights() {
                let input = if i > 0 {
                    outputs[i - 1].clone()
                } else {
                    inputs.to_owned()
                };
                if self.delta_weights[i].numel() == 0 {
                    self.delta_weights[i] = Tensor::zeros(&[input.numel(), gradient.numel()]);
                }
                if self.delta_biases[i].numel() == 0 {
                    self.delta_biases[i] = Tensor::zeros(&[gradient.numel()]);
                }
                self.delta_weights[i] += &input.outer(&gradient);
                self.delta_biases[i] += &gradient;
            }
            if i != 0 {
                gradient = layer.backward(&gradient.as_ref(), &outputs[i].as_ref());
            }
        }

        outputs.last().unwrap().to_owned()
    }

    pub fn update(&mut self, learning_rate: f32) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if layer.has_weights() {
                layer.update(
                    &(&self.delta_weights[i] * learning_rate).as_ref(),
                    &(&self.delta_biases[i] * learning_rate).as_ref(),
                );
                self.delta_weights[i].fill_(0.0);
                self.delta_biases[i].fill_(0.0);
            }
        }
    }
}
