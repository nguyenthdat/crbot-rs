use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;

use crate::model::QNet;

pub fn logger_init() -> anyhow::Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

    let env_filter = EnvFilter::from_default_env().add_directive("crbot_rs=debug".parse().unwrap());

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer())
        .init();

    Ok(())
}

pub fn save_model<B: AutodiffBackend>(model: &QNet<B>, path: &str) -> anyhow::Result<()> {
    let rec = BinFileRecorder::<FullPrecisionSettings>::new();
    // extension is auto-chosen; this will write `<path>.bin`
    model.clone().save_file(path, &rec)?;
    Ok(())
}

pub fn load_model<B: AutodiffBackend>(
    model: QNet<B>,
    path: &str,
    device: &B::Device,
) -> anyhow::Result<QNet<B>> {
    let rec = BinFileRecorder::<FullPrecisionSettings>::new();
    Ok(model.load_file(path, &rec, device)?)
}
