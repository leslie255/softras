use glam::*;

mod key_code;
pub use key_code::*;

mod color;
use crate::color::*;

#[derive(Debug, Clone, Copy)]
pub struct FrameOutput<'game> {
    pub display_buffer: &'game [u8],
    pub display_width: u32,
    pub display_height: u32,

    pub overlay_text: &'game str,

    /// Whether the game window should "capture" the cursor.
    /// If `true`, the cursor should be hidden and locked if possible.
    /// If `false`, the cursor should behave normally.
    pub captures_cursor: bool,
}

/// The main game state struct.
pub struct Game {
    frame_buffer: Vec<RgbaPixel>,
    frame_width: u32,
    frame_height: u32,

    key_states: [bool; 256],
    cursor_position: Option<Vec2>,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Self {
        Self {
            frame_buffer: Vec::new(),
            frame_width: 0,
            frame_height: 0,
            key_states: [false; _],
            cursor_position: None,
        }
    }

    pub fn display_buffer(&self) -> &[u8] {
        bytemuck::cast_slice(&self.frame_buffer)
    }

    pub fn display_width(&self) -> u32 {
        self.frame_width
    }

    pub fn display_height(&self) -> u32 {
        self.frame_height
    }

    pub fn frame(&mut self) -> FrameOutput<'_> {
        self.draw_frame();
        FrameOutput {
            display_buffer: bytemuck::cast_slice(&self.frame_buffer),
            display_width: self.frame_width,
            display_height: self.frame_height,
            overlay_text: "SOFTRAS v0.0.0",
            captures_cursor: false,
        }
    }

    pub fn resize_display(&mut self, width: u32, height: u32) {
        self.frame_width = width;
        self.frame_height = height;
    }

    pub fn key_down(&mut self, key_code: KeyCode) {
        self.key_states[key_code as usize] = true;
    }

    pub fn key_up(&mut self, key_code: KeyCode) {
        self.key_states[key_code as usize] = false;
    }

    pub fn cursor_moved(&mut self, x: f32, y: f32) {
        self.cursor_position = Some(vec2(x, y));
    }

    pub fn cursor_entered(&mut self) {}

    pub fn cursor_left(&mut self) {
        self.cursor_position = None;
    }

    pub fn cursor_scrolled_lines(&mut self, x: f32, y: f32) {
        _ = x;
        _ = y;
    }

    pub fn cursor_scrolled_pixels(&mut self, x: f32, y: f32) {
        _ = x;
        _ = y;
    }

    pub fn focused(&mut self) {}

    pub fn unfocused(&mut self) {
        bytemuck::fill_zeroes(&mut self.key_states);
    }

    fn key_is_down(&self, key_code: KeyCode) -> bool {
        self.key_states[key_code as usize]
    }

    #[allow(dead_code)]
    fn super_is_down(&self) -> bool {
        self.key_is_down(KeyCode::SuperLeft) || self.key_is_down(KeyCode::SuperRight)
    }

    #[allow(dead_code)]
    fn alt_is_down(&self) -> bool {
        self.key_is_down(KeyCode::AltLeft) || self.key_is_down(KeyCode::AltRight)
    }

    #[allow(dead_code)]
    fn shift_is_down(&self) -> bool {
        self.key_is_down(KeyCode::ShiftLeft) || self.key_is_down(KeyCode::ShiftRight)
    }

    fn draw_frame(&mut self) {
        self.frame_buffer.resize(
            self.frame_width as usize * self.frame_height as usize,
            Srgba::from_hex(0xFFFFFFFF).into(),
        );

        let (color0, color1) = match self.super_is_down() {
            true => (Srgb::from_hex(0xFFFFFF), Srgb::from_hex(0x0000FF)),
            false => (Srgb::from_hex(0xFFFFFF), Srgb::from_hex(0x800080)),
        };
        for x in 0..self.frame_width {
            for y in 0..self.frame_height {
                let mut t = match self.shift_is_down() {
                    true => x as f32 / self.frame_width as f32,
                    false => y as f32 / self.frame_height as f32,
                };
                if self.alt_is_down() {
                    t = 1. - t;
                }
                let color = Srgb::lerp(color0, color1, t);
                let i_pixel = y as usize * self.frame_width as usize + x as usize;
                self.frame_buffer[i_pixel] = Srgba::from(color).into();
            }
        }
    }
}
