use crbot::util;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    util::logger_init()?;

    Ok(())
}
