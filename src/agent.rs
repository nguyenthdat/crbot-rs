use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Shape, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

use crate::model::{QNet, QNetConfig};

#[derive(Debug, Clone)]
pub struct Transition {
    pub s: Vec<f32>,
    pub a: usize,
    pub r: f32,
    pub s2: Vec<f32>,
    pub done: bool,
}

pub struct DqnAgent<B: AutodiffBackend> {
    pub model: QNet<B>,
    pub target: QNet<B>,
    pub optimizer: OptimizerAdaptor<burn::optim::Adam, QNet<B>, B>,
    pub lr: f64,

    state_dim: usize,
    action_dim: usize,

    memory: VecDeque<Transition>,
    mem_cap: usize,

    pub gamma: f32,
    pub epsilon: f32,
    pub epsilon_min: f32,
    pub epsilon_decay: f32,

    rng: StdRng,
    device: B::Device,
}

impl<B: AutodiffBackend> DqnAgent<B> {
    pub fn new(device: B::Device, state_dim: usize, action_dim: usize) -> Self {
        let cfg = QNetConfig {
            input_dim: state_dim,
            hidden: 64,
            output_dim: action_dim,
        };
        let model = cfg.init::<B>(&device);
        let target = cfg.init::<B>(&device);

        let optimizer = AdamConfig::new().init::<B, QNet<B>>(); // Optimizer adaptor over Autodiff backend. 

        Self {
            model,
            target,
            optimizer,
            lr: 1e-3,
            state_dim,
            action_dim,
            memory: VecDeque::with_capacity(10_000),
            mem_cap: 10_000,
            gamma: 0.95,
            epsilon: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.997,
            rng: StdRng::seed_from_u64(42),
            device,
        }
    }

    pub fn remember(&mut self, t: Transition) {
        if self.memory.len() == self.mem_cap {
            self.memory.pop_front();
        }
        self.memory.push_back(t);
    }

    pub fn act(&mut self, state: &[f32]) -> usize {
        if self.rng.random::<f32>() < self.epsilon {
            return self.rng.random_range(0..self.action_dim);
        }
        // tensor shape [1, state_dim]
        let x = Tensor::<B, 2>::from_data(
            TensorData::new(state.to_vec(), Shape::new([1, self.state_dim])),
            &self.device,
        );
        let q = self.model.forward(x); // [1, A]
        let (_, idx) = q.max_dim_with_indices(1); // [1,1] values + [1,1] indices
        idx.into_data().as_slice::<i32>().unwrap()[0] as usize
    }

    /// Hard update: target <- model (copy weights).
    pub fn update_target(&mut self) {
        let record = self.model.clone().into_record();
        self.target = self.target.clone().load_record(record);
    }

    pub fn replay(&mut self, batch_size: usize) {
        if self.memory.len() < batch_size {
            return;
        }

        // Sample a mini-batch
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.random_range(0..self.memory.len()))
            .collect();
        let mut states = Vec::with_capacity(batch_size * self.state_dim);
        let mut next_states = Vec::with_capacity(batch_size * self.state_dim);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);

        for &i in &indices {
            let t = &self.memory[i];
            states.extend_from_slice(&t.s);
            next_states.extend_from_slice(&t.s2);
            actions.push(t.a as i32);
            rewards.push(t.r);
            dones.push(if t.done { 1.0 } else { 0.0 });
        }

        let s = Tensor::<B, 2>::from_data(
            TensorData::new(states, Shape::new([batch_size, self.state_dim])),
            &self.device,
        );
        let s2 = Tensor::<B, 2>::from_data(
            TensorData::new(next_states, Shape::new([batch_size, self.state_dim])),
            &self.device,
        );
        let a_idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(actions, Shape::new([batch_size, 1])),
            &self.device,
        );
        let r = Tensor::<B, 1>::from_data(
            TensorData::new(rewards, Shape::new([batch_size])),
            &self.device,
        );
        let done = Tensor::<B, 1>::from_data(
            TensorData::new(dones, Shape::new([batch_size])),
            &self.device,
        );

        // Q(s, ·) and Q_target(s', ·)
        let q_all = self.model.forward(s); // [B, A]
        let q_next_all = self.target.forward(s2).detach(); // [B, A], detached so we don't backprop into target. (detach is provided by Burn’s tensor API). 

        // Gather Q(s, a) for each row
        let q_sa = q_all.gather(1, a_idx).squeeze(1); // [B], gather per-row along dim=1. 

        // max_a' Q_target(s', a')
        let q_next_max = q_next_all.max_dim(1).squeeze(1); // [B] (values). 

        // y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        let y = r + q_next_max
            * (1.0 - done)
            * Tensor::<B, 1>::full(Shape::new([batch_size]), self.gamma, &self.device);

        // MSE( Q(s,a), y )
        let loss = MseLoss::new().forward(q_sa, y, Reduction::Auto);

        // Backprop + optimizer step
        let grads = loss.backward();
        let grads = burn::optim::GradientsParams::from_grads::<B, _>(grads, &self.model); // associate grads with model params. 
        self.model = self.optimizer.step(self.lr, self.model.clone(), grads);

        // Epsilon decay
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
            if self.epsilon < self.epsilon_min {
                self.epsilon = self.epsilon_min;
            }
        }
    }
}
