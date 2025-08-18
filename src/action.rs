use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Actions {
    pub os_type: String,
    pub images_path: PathBuf,

    pub top_left_x: u32,
    pub top_left_y: u32,
    pub bottom_right_x: u32,
    pub bottom_right_y: u32,
    pub field_area: (u32, u32, u32, u32),

    pub window_width: u32,
    pub window_height: u32,
}

#[bon::bon]
impl Actions {
    #[builder]
    pub fn new() -> Self {
        todo!("Implement Actions::new()")
    }
}
