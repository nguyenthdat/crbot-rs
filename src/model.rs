use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct QNet<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct QNetConfig {
    pub input_dim: usize,
    pub hidden: usize,
    pub output_dim: usize,
}

impl QNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> QNet<B> {
        let l1 = LinearConfig::new(self.input_dim, self.hidden).init(device);
        let l2 = LinearConfig::new(self.hidden, self.output_dim).init(device);
        QNet {
            l1,
            l2,
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> QNet<B> {
    /// x: [batch, input_dim] -> [batch, output_dim]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.l1.forward(x);
        let x = self.relu.forward(x);
        self.l2.forward(x)
    }
}
