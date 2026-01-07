#![feature(iter_array_chunks, read_array, normalize_lexically)] // FIXME: use of unstable features

use std::{
    fmt::Write as _, io, path::{Path, PathBuf}, time::{Duration, Instant}
};

use derive_more::{Display, Error, From};
use glam::*;

mod key_code;
mod obj_file;
mod render;
mod respack;

pub use key_code::*;
use obj_file::*;
use render::*;
use respack::*;

/// Packs the needed resources from the directory located in `softras/res` into a one respack file.
///
/// # Arguments
///
/// * `res_dir` - the path to the `softras/res` directory
/// * `output` - the output file (e.g. `"resources.respack.bin"`)
pub fn pack_resources(res_dir: &str, output: &str) -> Result<(), PackResourceError> {
    fn pack(res_packer: &mut ResourcePacker, path: &str) -> Result<(), PackResourceError> {
        res_packer
            .append_file(path)
            .map_err(|error| PackResourceError::MissingResource {
                path: {
                    let root: &Path = res_packer.root_path().as_ref();
                    let subpath: &Path = path.as_ref();
                    root.join(subpath)
                },
                error,
            })
    }

    let mut res_packer = ResourcePacker::new(res_dir);
    pack(&mut res_packer, "models/teapot.obj")?;

    res_packer
        .finish_into_file(output)
        .map_err(|error| PackResourceError::OutputFileError {
            path: PathBuf::from(output),
            error,
        })?;

    Ok(())
}

#[derive(Debug, Display, Error)]
pub enum PackResourceError {
    #[display("cannot read file at {}: {error}", path.display())]
    MissingResource { path: PathBuf, error: io::Error },
    #[display("cannot write to output file at {:?}: {error}", path.display())]
    OutputFileError { path: PathBuf, error: io::Error },
}

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
    canvas: Canvas,
    overlay_text: String,
    key_states: [bool; 256],
    cursor_position: Option<Vec2>,
    fps_meter: FpsMeter,
    previous_frame_instant: Option<Instant>,
    is_paused: bool,
    camera: Camera,
    teapot_vertices: Vec<Vertex>,
}

#[derive(Debug, Display, Error, From)]
pub enum GameInitError {
    #[display("Malformatted respack: {_0}")]
    RespackReadError(RespackReadError),
    #[display("resource {path:?} not found")]
    MissingResourceItem { path: String },
}

impl Game {
    /* === Start of Public Interface === */

    /// Initialize a new `Game` instance.
    ///
    /// # Arguments
    ///
    /// * `respack_bytes` - the content of the packed respack file created by `pack_resources`
    pub fn new(respack_bytes: Vec<u8>) -> Result<Self, GameInitError> {
        fn get_resource<'a>(respack: &'a ResPack, path: &str) -> Result<&'a [u8], GameInitError> {
            respack
                .get(path)
                .ok_or_else(|| GameInitError::MissingResourceItem {
                    path: String::from(path),
                })
        }

        let respack = ResPack::from_vec(respack_bytes)?;

        let teapot_obj = get_resource(&respack, "models/teapot.obj")?;

        Ok(Self {
            canvas: Canvas::new(),
            overlay_text: String::new(),
            key_states: [false; _],
            cursor_position: None,
            fps_meter: FpsMeter::new(),
            previous_frame_instant: None,
            is_paused: false,
            camera: Camera {
                position: vec3(0., 1., 10.),
                pitch_degrees: 0.,
                yaw_degrees: 0.,
                fov_y_degrees: 90.,
            },
            teapot_vertices: load_obj(std::str::from_utf8(teapot_obj).unwrap()),
        })
    }

    pub fn display_buffer(&self) -> &[u8] {
        bytemuck::cast_slice(self.canvas.frame_buffer())
    }

    pub fn display_width(&self) -> u32 {
        self.canvas.width()
    }

    pub fn display_height(&self) -> u32 {
        self.canvas.height()
    }

    pub fn overlay_text(&self) -> &str {
        &self.overlay_text
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
            display_buffer: self.display_buffer(),
            display_width: self.display_width(),
            display_height: self.display_height(),
            overlay_text: self.overlay_text(),
            captures_cursor: !self.is_paused,
        }
    }

    pub fn notify_display_resize(&mut self, width: u32, height: u32) {
        self.canvas.resize(width, height);
    }

    pub fn notify_key_down(&mut self, key_code: KeyCode) {
        if key_code == KeyCode::Escape {
            self.is_paused = !self.is_paused;
        }
        self.key_states[key_code as usize] = true;
    }

    pub fn notify_key_up(&mut self, key_code: KeyCode) {
        self.key_states[key_code as usize] = false;
    }

    pub fn notify_cursor_down(&mut self, button: MouseButton) {
        _ = button;
    }

    pub fn notify_cursor_up(&mut self, button: MouseButton) {
        _ = button;
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
        let rate = if self.control_is_down() { 4. } else { 1. };
        let fov = &mut self.camera.fov_y_degrees;
        *fov = (*fov + rate * 0.1 * y * 14.).clamp(20., 160.);
    }

    pub fn notify_cursor_scrolled_pixels(&mut self, x: f32, y: f32) {
        _ = x;
        let rate = if self.control_is_down() { 4. } else { 1. };
        let fov = &mut self.camera.fov_y_degrees;
        *fov = (*fov + rate * 0.1 * y).clamp(20., 160.);
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
    fn control_is_down(&self) -> bool {
        self.key_is_down(KeyCode::ControlLeft) || self.key_is_down(KeyCode::ControlRight)
    }

    #[allow(dead_code)]
    fn shift_is_down(&self) -> bool {
        self.key_is_down(KeyCode::ShiftLeft) || self.key_is_down(KeyCode::ShiftRight)
    }

    fn update_overlay_text(&mut self) {
        self.overlay_text.clear();

        // Pause prompt.
        if self.is_paused {
            _ = writeln!(&mut self.overlay_text, "[ESC] PAUSED");
        }

        // Name and version.
        let crate_version = env!("CARGO_PKG_VERSION");
        if cfg!(debug_assertions) {
            _ = writeln!(
                &mut self.overlay_text,
                "SOFTWARE RASTERIZER v{crate_version} (DEBUG BUILD)"
            );
        } else {
            _ = writeln!(
                &mut self.overlay_text,
                "SOFTWARE RASTERIZER v{crate_version}"
            );
        }

        // Resolution.
        _ = writeln!(
            &mut self.overlay_text,
            "resolution: {}x{}",
            self.canvas.width(),
            self.canvas.height(),
        );

        // FPS.
        _ = write!(&mut self.overlay_text, "FPS: ");
        match self.fps_meter.fps() {
            Some(fps) => _ = write!(&mut self.overlay_text, "{:3.0}", fps),
            None => _ = write!(&mut self.overlay_text, "***"),
        };
        _ = write!(&mut self.overlay_text, ", avg over 12: ");
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
        if self.canvas.width() == 0 || self.canvas.height() == 0 {
            return;
        }
        self.canvas.clear(RgbaU8::from_hex(0x020202FF));

        let view = self.camera.view_matrix();
        let projection = self
            .camera
            .projection_matrix(self.canvas.width() as f32, self.canvas.height() as f32);
        let light_direction = view.transform_vector3(vec3(-1., -1., -1.)).normalize();

        let mut draw_teapot =
            |scale: f32, position: Vec3, rotation_degrees: f32, color: Rgb| -> () {
                let shader = DirectionalShadingShader {
                    color,
                    light_direction,
                    shading_intensity: 1.4,
                    highlightness: 0.7,
                };
                let model = Mat4::from_translation(position)
                    * Mat4::from_scale(vec3(scale, scale, scale))
                    * Mat4::from_rotation_y(rotation_degrees.to_degrees());
                for indices in self.teapot_vertices.iter().copied().array_chunks::<3>() {
                    draw_triangle(&mut self.canvas, view * model, projection, indices, &shader);
                }
            };
        draw_teapot(1., vec3(0., 0., 0.), 0., Rgb::from_hex(0xC0C0C0));
        draw_teapot(0.6, vec3(3., 0., 3.), 45., Rgb::from_hex(0xE0A080));
        draw_teapot(0.7, vec3(-3., 0., 4.), 225., Rgb::from_hex(0xA080E0));

        let mut draw_cube = |scale: f32, position: Vec3, rotation_degrees: f32, color: Rgb| -> () {
            let shader = DirectionalShadingShader {
                color,
                light_direction,
                shading_intensity: 1.,
                ..Default::default()
            };
            let model = Mat4::from_translation(position)
                * Mat4::from_scale(vec3(scale, scale, scale))
                * Mat4::from_rotation_y(rotation_degrees.to_degrees());
            for indices in Self::CUBE_INDICIES {
                let vertices = indices.map(|i| Self::CUBE_VERTICES[i as usize]);
                draw_triangle(
                    &mut self.canvas,
                    view * model,
                    projection,
                    vertices,
                    &shader,
                );
            }
        };
        draw_cube(1., vec3(6., 0., 2.), 30., Rgb::from_hex(0x008080));
        draw_cube(1.5, vec3(0., 0., 6.), 45., Rgb::from_hex(0x800080));

        #[rustfmt::skip]
        let ground_vertices = [
            Vertex::new([1., 0., 0.], [0., 1.], [0., 1., 0.]),
            Vertex::new([0., 0., 0.], [1., 1.], [0., 1., 0.]),
            Vertex::new([0., 0., 1.], [1., 0.], [0., 1., 0.]),
            Vertex::new([1., 0., 1.], [0., 0.], [0., 1., 0.]),
        ];
        let ground_indices = [[0u16, 1, 2], [2, 3, 0]];
        let shader = DirectionalShadingShader {
            color: Rgb::from_hex(0x101820),
            light_direction: vec3(-1., -1., -1.).normalize(),
            ..Default::default()
        };
        let ground_size = 100.0f32;
        let model = Mat4::from_translation(0.5 * vec3(-ground_size, 0., -ground_size))
            * Mat4::from_scale(vec3(ground_size, 0., ground_size));
        for indices in ground_indices {
            draw_triangle(
                &mut self.canvas,
                view * model,
                projection,
                indices.map(|i| ground_vertices[i as usize]),
                &shader,
            );
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
        if self.key_is_down(KeyCode::ControlLeft) && movement.z >= 0. {
            movement.z *= 2.;
        }
        if self.key_is_down(KeyCode::F3) {
            movement *= 32.;
        }
        if self.alt_is_down() {
            movement /= 8.;
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

#[derive(Debug, Clone, Copy)]
struct FpsMeter {
    frame_times: [f64; 12],
    cursor: usize,
    average_frame_time: Option<f64>,
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
            average_frame_time: None,
        }
    }

    fn new_frame_time(&mut self, frame_time: f64) {
        let i = self.cursor;
        self.cursor += 1;
        self.cursor %= self.frame_times.len();
        self.frame_times[i] = frame_time;
        let frame_time = self.frame_times.iter().sum::<f64>() / (self.frame_times.len() as f64);
        match &mut self.average_frame_time {
            Some(frame_time_) => *frame_time_ = frame_time,
            frame_time_ @ None if self.cursor == 0 => *frame_time_ = Some(frame_time),
            None => (),
        }
    }

    fn average_frame_time(&self) -> Option<f64> {
        self.average_frame_time
    }

    fn average_fps(&self) -> Option<f64> {
        self.average_frame_time().map(|f| 1. / f)
    }

    fn fps(&self) -> Option<f64> {
        if self.average_frame_time.is_some() || self.cursor >= 1 {
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
