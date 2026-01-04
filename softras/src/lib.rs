use std::{
    fmt::Write as _,
    time::{Duration, Instant, SystemTime},
};

use bytemuck::{Pod, Zeroable};
use glam::*;

mod key_code;
pub use key_code::*;

mod color;
use crate::color::*;

mod shader;
use crate::shader::*;

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
    overlay_text: String,

    key_states: [bool; 256],
    cursor_position: Option<Vec2>,

    fps_meter: FpsMeter,

    camera: Camera,
    previous_frame_instant: Option<Instant>,
    is_paused: bool,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    /* === Start of Public Interface === */

    pub fn new() -> Self {
        Self {
            frame_buffer: Vec::new(),
            depth_buffer: Vec::new(),
            frame_width: 0,
            frame_height: 0,
            overlay_text: String::new(),
            key_states: [false; _],
            cursor_position: None,
            fps_meter: FpsMeter::new(),
            camera: Camera {
                position: vec3(0., 0., 20.),
                pitch_degrees: 0.,
                yaw_degrees: 0.,
                fov_y_degrees: 90.,
            },
            previous_frame_instant: None,
            is_paused: false,
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
        let before_frame = Instant::now();
        if let Some(previous_frame_instant) = self.previous_frame_instant {
            self.move_player(before_frame.duration_since(previous_frame_instant));
        }
        self.previous_frame_instant = Some(before_frame);

        self.draw();

        let after_frame = Instant::now();
        let frame_time = (after_frame - before_frame).as_secs_f64();
        self.fps_meter.new_frame_time(frame_time);

        self.update_overlay_text();

        FrameOutput {
            display_buffer: bytemuck::cast_slice(&self.frame_buffer),
            display_width: self.frame_width,
            display_height: self.frame_height,
            overlay_text: &self.overlay_text,
            captures_cursor: !self.is_paused,
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

    pub fn notify_cursor_moved(&mut self, x: f32, y: f32, dx: f32, dy: f32) {
        self.cursor_position = Some(vec2(x, y));
        self.rotation_player(vec2(dx, dy));
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

    /* === End of Public Interface === */

    const CUBE_VERTICES: [Vertex; 24] = [
        // South
        Vertex::new([0., 0., 1.], [0., 1.], [0., 0., 1.]),
        Vertex::new([1., 0., 1.], [1., 1.], [0., 0., 1.]),
        Vertex::new([1., 1., 1.], [1., 0.], [0., 0., 1.]),
        Vertex::new([0., 1., 1.], [0., 0.], [0., 0., 1.]),
        // North
        Vertex::new([0., 0., 0.], [1., 1.], [0., 0., -1.]),
        Vertex::new([0., 1., 0.], [1., 0.], [0., 0., -1.]),
        Vertex::new([1., 1., 0.], [0., 0.], [0., 0., -1.]),
        Vertex::new([1., 0., 0.], [0., 1.], [0., 0., -1.]),
        // East
        Vertex::new([1., 0., 0.], [1., 1.], [1., 0., 0.]),
        Vertex::new([1., 1., 0.], [1., 0.], [1., 0., 0.]),
        Vertex::new([1., 1., 1.], [0., 0.], [1., 0., 0.]),
        Vertex::new([1., 0., 1.], [0., 1.], [1., 0., 0.]),
        // West
        Vertex::new([0., 1., 0.], [0., 0.], [-1., 0., 0.]),
        Vertex::new([0., 0., 0.], [0., 1.], [-1., 0., 0.]),
        Vertex::new([0., 0., 1.], [1., 1.], [-1., 0., 0.]),
        Vertex::new([0., 1., 1.], [1., 0.], [-1., 0., 0.]),
        // Up
        Vertex::new([1., 1., 0.], [0., 1.], [0., 1., 0.]),
        Vertex::new([0., 1., 0.], [1., 1.], [0., 1., 0.]),
        Vertex::new([0., 1., 1.], [1., 0.], [0., 1., 0.]),
        Vertex::new([1., 1., 1.], [0., 0.], [0., 1., 0.]),
        // Down
        Vertex::new([0., 0., 0.], [0., 1.], [0., -1., 0.]),
        Vertex::new([1., 0., 0.], [1., 1.], [0., -1., 0.]),
        Vertex::new([1., 0., 1.], [1., 0.], [0., -1., 0.]),
        Vertex::new([0., 0., 1.], [0., 0.], [0., -1., 0.]),
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

    fn update_overlay_text(&mut self) {
        self.overlay_text.clear();

        // Name and Version.
        let crate_version = env!("CARGO_PKG_VERSION");
        _ = writeln!(
            &mut self.overlay_text,
            "SOFTWARE RASTERIZER v{crate_version}"
        );

        // Resolution.
        _ = writeln!(
            &mut self.overlay_text,
            "resolution: {}x{}",
            self.frame_width, self.frame_height,
        );

        // FPS.
        _ = write!(&mut self.overlay_text, "FPS: ");
        match self.fps_meter.fps() {
            Some(fps) => _ = write!(&mut self.overlay_text, "{:3.0}", fps),
            None => _ = write!(&mut self.overlay_text, "***"),
        };
        _ = write!(&mut self.overlay_text, ", avg over 48: ");
        match self.fps_meter.average_fps() {
            Some(avarage_fps) => _ = writeln!(&mut self.overlay_text, "{:3.0}", avarage_fps),
            None => _ = writeln!(&mut self.overlay_text, "***"),
        };

        // Camera position/direction.
        _ = writeln!(
            &mut self.overlay_text,
            "camera x/y/z: {:.04}/{:.04}/{:.04}",
            self.camera.position.x, self.camera.position.y, self.camera.position.z,
        );

        // Camera position/direction.
        _ = writeln!(
            &mut self.overlay_text,
            "camera pitch/yaw: {:.04}/{:.04}",
            self.camera.pitch_degrees, self.camera.yaw_degrees,
        );

        // Camera position/direction.
        _ = writeln!(
            &mut self.overlay_text,
            "camera FOV: {:.04}",
            self.camera.fov_y_degrees,
        );
    }

    fn draw(&mut self) {
        let background_color = Rgb::from_hex(0x040404);
        let n_pixels = self.frame_width as usize * self.frame_height as usize;
        self.frame_buffer.resize(n_pixels, RgbaU8::zeroed());
        self.frame_buffer.fill(background_color.into());
        self.depth_buffer.resize(n_pixels, 0.0f32);
        self.depth_buffer.fill(f32::INFINITY);
        let t = (SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0.0f64, |duration| duration.as_secs_f64())
            % 86400.) as f32;
        let view = self.camera.view_matrix();
        let projection = self
            .camera
            .projection_matrix(self.frame_width as f32, self.frame_height as f32);

        let cubes: [(Vec3, f32, Rgb, f32); _] = [
            // position, size, color, rotation speed
            (vec3(-15., 0., 0.), 3., Rgb::from_hex(0x008080), -2.0f32),
            (vec3(-5., 0., 0.), 5., Rgb::from_hex(0xA06000), 0.5f32),
            (vec3(5., 0., 0.), 4., Rgb::from_hex(0x00A060), -1.0f32),
            (vec3(15., 0., 0.), 3., Rgb::from_hex(0x800080), 1.5f32),
        ];

        for y_repeat in -1i32..=2i32 {
            for z_repeat in -3i32..=0i32 {
                for (position, size, color, rotation_speed) in cubes {
                    let position = Vec3 {
                        y: y_repeat as f32 * 10.,
                        z: z_repeat as f32 * 10.,
                        ..position
                    };
                    let rotation_speed = match z_repeat % 2 == 0 {
                        true => rotation_speed,
                        false => -rotation_speed,
                    };
                    let size_vec3 = vec3(size, size, size);
                    // let shader = DepthVisualizationShader {
                    //     z_min: 0.,
                    //     z_max: 1.,
                    // };
                    let shader = Basic3dShader {
                        color,
                        ..Default::default()
                    };
                    let model = Mat4::from_translation(position)
                        // * Mat4::from_rotation_x(t * rotation_speed)
                        // * Mat4::from_rotation_z(t * rotation_speed)
                        * Mat4::from_translation(-0.5 * size_vec3)
                        * Mat4::from_scale(size_vec3);
                    let model_view = view * model;
                    for indices in Self::CUBE_INDICIES {
                        let triangle = indices.map(|i| Self::CUBE_VERTICES[i as usize]);
                        self.draw_triangle(triangle, projection, model_view, &shader);
                    }
                }
            }
        }
    }

    fn draw_triangle<S: Shader + ?Sized>(
        &mut self,
        triangle_local: [Vertex; 3],
        projection: Mat4,
        model_view: Mat4,
        shader: &S,
    ) {
        let mvp = projection * model_view;
        let triangle = triangle_local.map(|vertex| {
            let clip = mvp * vertex.position.extend(1.);
            Vertex {
                position: clip.xyz() / clip.w,
                normal: (model_view.transform_vector3(vertex.normal) * clip.w)
                    .normalize_or(vec3(1., 0., 0.)),
                ..vertex
            }
        });
        if triangle.map(|vertex| vertex.position.z < 0.) == [true; 3] {
            return;
        }
        if triangle.map(|vertex| vertex.position.z > 1.) == [true; 3] {
            return;
        }
        if !is_clockwise_winding(triangle.map(|vertex| vertex.position.xy())) {
            return;
        }
        let [x_min, x_max, y_min, y_max]: [u32; 4] = {
            let [p0, p1, p2] = triangle.map(|vertex| vertex.position.xy());
            let x_min_ndc = p0.x.min(p1.x).min(p2.x);
            let x_max_ndc = p0.x.max(p1.x).max(p2.x);
            let y_min_ndc = p0.y.min(p1.y).min(p2.y);
            let y_max_ndc = p0.y.max(p1.y).max(p2.y);
            let width = self.frame_width;
            let height = self.frame_height;
            [
                (ndc_to_pixel_x(x_min_ndc, width).floor().max(0.) as u32).min(width - 1),
                (ndc_to_pixel_x(x_max_ndc, width).ceil().max(0.) as u32).min(width - 1),
                (ndc_to_pixel_y(y_max_ndc, height).floor().max(0.) as u32).min(height - 1),
                (ndc_to_pixel_y(y_min_ndc, height).ceil().max(0.) as u32).min(height - 1),
            ]
        };
        for x_pixel in x_min..=x_max {
            for y_pixel in y_min..=y_max {
                let i_pixel = y_pixel as usize * self.frame_width as usize + x_pixel as usize;
                let ndc = vec2(
                    pixel_to_ndc_x(x_pixel, self.frame_width),
                    pixel_to_ndc_y(y_pixel, self.frame_height),
                );
                let rasterize_result = rasterize(triangle.map(|vertex| vertex.position.xy()), ndc);
                let Some(weights) = rasterize_result else {
                    continue;
                };
                let z = triangular_interpolate(weights, triangle.map(|v| v.position.z));
                if !(-1.0..1.0).contains(&ndc.x)
                    || !(-1.0..1.0).contains(&ndc.y)
                    || !(0.0..1.0).contains(&z)
                {
                    continue;
                }
                let depth = &mut self.depth_buffer[i_pixel];
                if *depth <= z {
                    continue;
                }
                *depth = z;
                let fragment_input = FragmentInput {
                    position: vec3(ndc.x, ndc.y, z),
                    depth: z,
                    uv: vec2(
                        triangular_interpolate(weights, triangle.map(|v| v.uv.x)),
                        triangular_interpolate(weights, triangle.map(|v| v.uv.y)),
                    ),
                    normal: vec3(
                        triangular_interpolate(weights, triangle.map(|v| v.normal.x)),
                        triangular_interpolate(weights, triangle.map(|v| v.normal.y)),
                        triangular_interpolate(weights, triangle.map(|v| v.normal.z)),
                    ),
                };
                let fragment_result = shader.fragment(fragment_input);
                self.frame_buffer[i_pixel] = RgbaU8::from(fragment_result);
            }
        }
    }

    fn move_player(&mut self, frame_duration: Duration) {
        if self.is_paused {
            return;
        }
        let mut movement = vec3(0., 0., 0.);
        if self.key_is_down(KeyCode::KeyW) {
            movement.z += 1.;
        }
        if self.key_is_down(KeyCode::KeyS) {
            movement.z -= 1.;
        }
        if self.key_is_down(KeyCode::KeyA) {
            movement.x -= 1.;
        }
        if self.key_is_down(KeyCode::KeyD) {
            movement.x += 1.;
        }
        if self.key_is_down(KeyCode::Space) {
            movement.y += 1.;
        }
        if self.key_is_down(KeyCode::KeyR) || self.key_is_down(KeyCode::ShiftLeft) {
            movement.y -= 1.;
        }
        movement = 8. * movement.normalize();
        if movement.x.is_nan() | movement.y.is_nan() | movement.z.is_nan() {
            return;
        }
        if self.key_is_down(KeyCode::ControlLeft) && movement.z > 0. {
            movement.z *= 2.;
        }
        if self.key_is_down(KeyCode::F3) {
            movement *= 32.;
        }
        movement *= frame_duration.as_secs_f32();

        self.camera.move_(movement);
    }

    fn rotation_player(&mut self, delta: Vec2) {
        if self.is_paused {
            return;
        }
        let sensitivity = 1. / 11.;
        self.camera.yaw_degrees += sensitivity * delta.x;
        self.camera.pitch_degrees -= sensitivity * delta.y;
        self.camera.pitch_degrees = self.camera.pitch_degrees.clamp(-90. + 0.0001, 90. - 0.0001);
        if self.camera.yaw_degrees <= -180. {
            self.camera.yaw_degrees += 360.;
        } else if self.camera.yaw_degrees >= 180. {
            self.camera.yaw_degrees -= 360.;
        }
    }
}

fn triangular_interpolate([w0, w1, w2]: [f32; 3], [x0, x1, x2]: [f32; 3]) -> f32 {
    w0 * x0 + w1 * x1 + w2 * x2
}

fn ndc_to_pixel_x(x_ndc: f32, width: u32) -> f32 {
    let width_f = width as f32;
    0.5 * width_f * (x_ndc + 1.)
}

fn ndc_to_pixel_y(y_ndc: f32, height: u32) -> f32 {
    let height_f = height as f32;
    -0.5 * height_f * (y_ndc - 1.)
}

fn pixel_to_ndc_x(x_pixel: u32, width: u32) -> f32 {
    let x_pixel_f = x_pixel.min(width - 1) as f32;
    let width_f = width as f32;
    2. * x_pixel_f / width_f - 1.
}

fn pixel_to_ndc_y(y_pixel: u32, height: u32) -> f32 {
    let y_pixel_f = y_pixel.min(height - 1) as f32;
    let height_f = height as f32;
    -2. * y_pixel_f / height_f + 1.
}

fn is_clockwise_winding([a, b, c]: [Vec2; 3]) -> bool {
    fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
        0.5 * (c - a).dot((b - a).perp())
    }
    signed_area(a, b, c) >= 0.
}

/// If point `p` is inside the triangle formed by XP components of points `a`, `b`, and `c`,
/// returns the weights of `a`, `b`, and `c` for triangular-interpolation.
fn rasterize([a, b, c]: [Vec2; 3], p: Vec2) -> Option<[f32; 3]> {
    fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
        0.5 * (c - a).dot((b - a).perp())
    }
    let area_bcp = signed_area(b, c, p);
    let area_cap = signed_area(c, a, p);
    let area_abp = signed_area(a, b, p);
    let area_total = area_bcp + area_cap + area_abp;
    if (area_bcp > 0.) == (area_cap > 0.) && (area_cap > 0.) == (area_abp > 0.) {
        // Inside.
        let inv_area_total = 1. / area_total;
        Some([
            inv_area_total * area_bcp, // weight of A
            inv_area_total * area_cap, // weight of B
            inv_area_total * area_abp, // weight of C
        ])
    } else {
        // Outside.
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    uv: Vec2,
    normal: Vec3,
}

impl Vertex {
    const fn new(
        [x, y, z]: [f32; 3],
        [u, v]: [f32; 2],
        [x_normal, y_normal, z_normal]: [f32; 3],
    ) -> Self {
        Self {
            position: vec3(x, y, z),
            uv: vec2(u, v),
            normal: vec3(x_normal, y_normal, z_normal),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Camera {
    position: Vec3,
    pitch_degrees: f32,
    yaw_degrees: f32,
    fov_y_degrees: f32,
}

impl Camera {
    pub fn direction(self) -> Vec3 {
        let pitch = self.pitch_degrees.to_radians();
        let yaw = self.yaw_degrees.to_radians();
        vec3(
            f32::sin(yaw) * f32::cos(pitch),
            f32::sin(pitch),
            -f32::cos(yaw) * f32::cos(pitch),
        )
    }

    pub fn move_(&mut self, delta: Vec3) {
        let direction = self.direction();
        let forward = direction.with_y(0.).normalize();
        let up = vec3(0., 1., 0.);
        let right = forward.cross(up).normalize();
        let forward_scaled = delta.z * forward;
        let right_scaled = delta.x * right;
        let up_scaled = delta.y * up;
        self.position += forward_scaled + right_scaled + up_scaled;
    }

    pub fn view_matrix(self) -> Mat4 {
        Mat4::look_to_rh(
            self.position,    // eye
            self.direction(), // dir
            vec3(0., 1., 0.), // up
        )
    }

    pub fn projection_matrix(self, frame_width: f32, frame_height: f32) -> Mat4 {
        let aspect = frame_width / frame_height;
        let fovy = self.fov_y_degrees.to_radians();
        let near = 0.1;
        let far = 100.;

        let f = 1. / f32::tan(fovy / 2.);

        let c0r0 = f / aspect;
        let c0r1 = 0.;
        let c0r2 = 0.;
        let c0r3 = 0.;

        let c1r0 = 0.;
        let c1r1 = f;
        let c1r2 = 0.;
        let c1r3 = 0.;

        let c2r0 = 0.;
        let c2r1 = 0.;
        let c2r2 = (far + near) / (near - far);
        let c2r3 = -1.;

        let c3r0 = 0.;
        let c3r1 = 0.;
        let c3r2 = (2. * far * near) / (near - far);
        let c3r3 = 0.;

        mat4(
            vec4(c0r0, c0r1, c0r2, c0r3),
            vec4(c1r0, c1r1, c1r2, c1r3),
            vec4(c2r0, c2r1, c2r2, c2r3),
            vec4(c3r0, c3r1, c3r2, c3r3),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct FpsMeter {
    frame_times: [f64; 48],
    cursor: usize,
    frame_time: Option<f64>,
}

impl Default for FpsMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl FpsMeter {
    const fn new() -> Self {
        Self {
            frame_times: [0.; _],
            cursor: 0,
            frame_time: None,
        }
    }

    fn new_frame_time(&mut self, frame_time: f64) {
        let i = self.cursor;
        self.cursor += 1;
        self.cursor %= self.frame_times.len();
        self.frame_times[i] = frame_time;
        let frame_time = self.frame_times.iter().sum::<f64>() / (self.frame_times.len() as f64);
        match &mut self.frame_time {
            Some(frame_time_) => *frame_time_ = frame_time,
            frame_time_ @ None if self.cursor == 0 => *frame_time_ = Some(frame_time),
            None => (),
        }
    }

    fn average_frame_time(&self) -> Option<f64> {
        self.frame_time
    }

    fn average_fps(&self) -> Option<f64> {
        self.average_frame_time().map(|f| 1. / f)
    }

    fn fps(&self) -> Option<f64> {
        if self.frame_time.is_some() || self.cursor >= 1 {
            let i = self
                .cursor
                .checked_sub(1)
                .unwrap_or(self.frame_times.len() - 1);
            Some(1. / self.frame_times[i])
        } else {
            None
        }
    }
}
