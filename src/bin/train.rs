use burn::backend::Autodiff;
use burn::backend::Cuda;
use burn::backend::cuda::CudaDevice;
use burn::prelude::*;

use crbot::agent::DqnAgent;
use crbot::agent::Transition;
use crbot::util;

type AD = Autodiff<Cuda>;

fn main() {
    dotenvy::dotenv().ok();
    util::logger_init().expect("Failed to initialize logger");
    // ----- CUDA device -----
    // Choose GPU 0; change if you have multiple GPUs.
    let device = CudaDevice::new(0);

    // ----- Your environment sizes -----
    let state_dim = 1 + 2 * (10 + 10); // match your env if you reuse it
    let action_dim = 4 * 18 * 28 + 1; // or whatever you enumerate (cards x grid + no-op)

    let mut agent: DqnAgent<AD> = DqnAgent::new(device, state_dim, action_dim);

    // ----- (Pseudo) training loop -----
    let episodes = 10_000usize;
    let batch_size = 32usize;

    for ep in 0..episodes {
        // TODO: plug in your real env reset -> Vec<f32>
        let mut state = vec![0.0f32; state_dim];

        let mut done = false;
        let mut total = 0.0f32;

        // (Replace this with: while !done { ... step env ... })
        for _t in 0..200 {
            let action = agent.act(&state);

            // TODO: env.step(action) -> (next_state, reward, done)
            let next_state = vec![0.0f32; state_dim];
            let reward = 0.0f32;
            done = false;

            agent.remember(Transition {
                s: state.clone(),
                a: action,
                r: reward,
                s2: next_state.clone(),
                done,
            });
            agent.replay(batch_size);

            state = next_state;
            total += reward;

            if done {
                break;
            }
        }

        if ep % 10 == 0 {
            agent.update_target();
            println!("Episode {ep} | epsilon={:.3}", agent.epsilon);
        }
    }
}
