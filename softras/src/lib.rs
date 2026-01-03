use std::time::SystemTime;

use bytemuck::Zeroable as _;
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
    frame_buffer: Vec<RgbaU8>,
    depth_buffer: Vec<f32>,
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
            depth_buffer: Vec::new(),
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
        self.draw();
        FrameOutput {
            display_buffer: bytemuck::cast_slice(&self.frame_buffer),
            display_width: self.frame_width,
            display_height: self.frame_height,
            overlay_text: "SOFTRAS v0.0.0",
            captures_cursor: false,
        }
    }

    pub fn notify_display_resize(&mut self, width: u32, height: u32) {
        self.frame_width = width;
        self.frame_height = height;
    }

    pub fn notify_key_down(&mut self, key_code: KeyCode) {
        self.key_states[key_code as usize] = true;
    }

    pub fn notify_key_up(&mut self, key_code: KeyCode) {
        self.key_states[key_code as usize] = false;
    }

    pub fn notify_cursor_moved(&mut self, x: f32, y: f32) {
        self.cursor_position = Some(vec2(x, y));
    }

    pub fn notify_cursor_entered(&mut self) {}

    pub fn notify_cursor_left(&mut self) {
        self.cursor_position = None;
    }

    pub fn notify_cursor_scrolled_lines(&mut self, x: f32, y: f32) {
        _ = x;
        _ = y;
    }

    pub fn notify_cursor_scrolled_pixels(&mut self, x: f32, y: f32) {
        _ = x;
        _ = y;
    }

    pub fn notify_focused(&mut self) {}

    pub fn notify_unfocused(&mut self) {
        bytemuck::fill_zeroes(&mut self.key_states);
    }

    const CUBE_VERTICES: [Vec3; 24] = [
        // South
        vec3(0., 0., 1.),
        vec3(1., 0., 1.),
        vec3(1., 1., 1.),
        vec3(0., 1., 1.),
        // North
        vec3(0., 0., 0.),
        vec3(0., 1., 0.),
        vec3(1., 1., 0.),
        vec3(1., 0., 0.),
        // East
        vec3(1., 0., 0.),
        vec3(1., 1., 0.),
        vec3(1., 1., 1.),
        vec3(1., 0., 1.),
        // West
        vec3(0., 1., 0.),
        vec3(0., 0., 0.),
        vec3(0., 0., 1.),
        vec3(0., 1., 1.),
        // Up
        vec3(1., 1., 0.),
        vec3(0., 1., 0.),
        vec3(0., 1., 1.),
        vec3(1., 1., 1.),
        // Down
        vec3(0., 0., 0.),
        vec3(1., 0., 0.),
        vec3(1., 0., 1.),
        vec3(0., 0., 1.),
    ];

    #[rustfmt::skip]
    const CUBE_INDICIES: [[u16; 3]; 12] = [
        /* South */ [0, 1, 2], [2, 3, 0],
        /* North */ [4, 5, 6], [6, 7, 4],
        /* East  */ [8, 9, 10], [10, 11, 8],
        /* West  */ [12, 13, 14], [14, 15, 12],
        /* Up    */ [16, 17, 18], [18, 19, 16],
        /* Down  */ [20, 21, 22], [22, 23, 20],
    ];

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

    fn draw(&mut self) {
        let background_color = Rgb::from_hex(0x040404);
        let n_pixels = self.frame_width as usize * self.frame_height as usize;
        self.frame_buffer.resize(n_pixels, RgbaU8::zeroed());
        self.frame_buffer.fill(background_color.into());
        self.depth_buffer.resize(n_pixels, 0.0f32);
        self.depth_buffer.fill(f32::INFINITY);
        let palatte = [
            Rgb::from_hex(0xFF0000),
            Rgb::from_hex(0x808000),
            Rgb::from_hex(0x00FF00),
            Rgb::from_hex(0x008080),
            Rgb::from_hex(0x0000FF),
            Rgb::from_hex(0x800080),
        ];
        let t = (SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0.0f64, |duration| duration.as_secs_f64())
            % 86400.) as f32;
        let size = vec3(500., 500., 500.);
        let model = Mat4::from_rotation_x(t)
            * Mat4::from_rotation_y(t)
            * Mat4::from_rotation_z(t)
            * Mat4::from_translation(-0.5 * size)
            * Mat4::from_scale(size);
        for (&indices, &color) in Self::CUBE_INDICIES.iter().zip(palatte.iter().cycle()) {
            let triangle_local = indices.map(|i| Self::CUBE_VERTICES[i as usize]);
            let triangle_world = triangle_local.map(|v| model.transform_point3(v));
            self.draw_triangle(triangle_world, color);
        }
    }

    fn draw_triangle(&mut self, triangle: [Vec3; 3], color: Rgb) {
        for x_pixel in 0..self.frame_width {
            for y_pixel in 0..self.frame_height {
                let i_pixel = y_pixel as usize * self.frame_width as usize + x_pixel as usize;
                let coord = vec2(
                    x_pixel as f32 - 0.5 * (self.frame_width as f32), //
                    -(y_pixel as f32) + 0.5 * (self.frame_height as f32), //
                );
                if let Some(this_depth) = rasterize(triangle, coord) {
                    let depth = &mut self.depth_buffer[i_pixel];
                    if *depth > this_depth {
                        self.frame_buffer[i_pixel] = RgbaU8::from(color);
                        *depth = this_depth;
                    }
                };
            }
        }
    }
}

/// If point `p` is inside the triangle formed by XP components of points `a`, `b`, and `c`,
/// returns the triangular-interpolated value between `a.z`, `b.z`, and `c.z` (returns
/// `None` otherwise).
fn rasterize([a, b, c]: [Vec3; 3], p: Vec2) -> Option<f32> {
    fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
        0.5 * (c - a).dot((b - a).perp())
    }
    let (a_z, a) = (a.z, a.xy());
    let (b_z, b) = (b.z, b.xy());
    let (c_z, c) = (c.z, c.xy());
    let area_bcp = signed_area(b, c, p);
    let area_cap = signed_area(c, a, p);
    let area_abp = signed_area(a, b, p);
    let sign_bcp = area_bcp >= 0.;
    let sign_cap = area_cap >= 0.;
    let sign_abp = area_abp >= 0.;
    let area_total = area_bcp + area_cap + area_abp;
    if sign_bcp && sign_cap && sign_abp && area_total > f32::EPSILON {
        // Inside.
        let inv_area_total = 1. / area_total;
        Some(
            inv_area_total * area_bcp * a_z
                + inv_area_total * area_cap * b_z
                + inv_area_total * area_abp * c_z,
        )
    } else {
        // Outside.
        None
    }
}
