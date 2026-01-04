use std::sync::Arc;

use muilib::{
    Canvas as _, EventLoopExt as _, RectSize, Srgb, Srgba,
    cgmath::*,
    winit::{
        application::ApplicationHandler,
        event::{DeviceEvent, DeviceId, MouseScrollDelta, WindowEvent},
        event_loop::{ActiveEventLoop, EventLoop},
        keyboard::PhysicalKey,
        window::{CursorGrabMode, Window, WindowId},
    },
};

mod key_code;

fn main() {
    env_logger::init();

    let resources = muilib::AppResources::new("muilib_backend/res".into());
    let event_loop = EventLoop::builder().build().unwrap();
    event_loop
        .run_lazy_initialized_app::<App, _>(&resources)
        .unwrap();
}

struct App<'cx> {
    window: Arc<Window>,
    uicx: muilib::UiContext<'cx>,
    canvas: muilib::WindowCanvas<'cx>,
    game: softras::Game,
    image_view: muilib::ImageView,
    overlay_text_view: muilib::TextView<'cx>,
    cursor_captured: bool,
    cursor_position: Option<Vector2<f32>>,
}

impl<'cx> App<'cx> {
    fn draw_frame(&mut self, canvas: muilib::CanvasRef) {
        let mut render_pass = self
            .uicx
            .begin_render_pass(&canvas, Srgb::from_hex(0x000000));

        // Draw scene.
        let frame_output = self.game.frame();
        let frame_image = muilib::ImageRef {
            size: RectSize::new(frame_output.display_width, frame_output.display_height),
            format: muilib::wgpu::TextureFormat::Rgba8Unorm,
            data: frame_output.display_buffer,
        };
        if frame_image.size.width > 0 && frame_image.size.height > 0 {
            let frame_texture = self.uicx.create_texture(frame_image);
            self.image_view.set_texture(frame_texture);
            self.uicx
                .prepare_view_bounded(&canvas, canvas.bounds(), &mut self.image_view);
            self.uicx.draw_view(&mut render_pass, &self.image_view);
        }

        // Draw overlay text.
        self.overlay_text_view
            .set_text(String::from(frame_output.overlay_text));
        self.uicx
            .prepare_view(&canvas, point2(4., 4.), &mut self.overlay_text_view);
        self.uicx
            .draw_view(&mut render_pass, &self.overlay_text_view);

        // Capture cursor if requested.
        let requested_captures_cursor = frame_output.captures_cursor;
        if !self.cursor_captured && requested_captures_cursor {
            self.capture_cursor();
        } else if self.cursor_captured && !requested_captures_cursor {
            self.uncapture_cursor();
        }
        self.cursor_captured = requested_captures_cursor;
    }

    fn capture_cursor(&self) {
        match self.window.set_cursor_grab(CursorGrabMode::Locked) {
            Ok(_) => self.window.set_cursor_visible(false),
            Err(error) => log::error!("unable to lock cursor: {error}"),
        }
    }

    fn uncapture_cursor(&self) {
        match self.window.set_cursor_grab(CursorGrabMode::None) {
            Ok(_) => self.window.set_cursor_visible(true),
            Err(error) => log::error!("unable to unlock cursor: {error}"),
        }
    }
}

impl<'cx> muilib::LazyApplicationHandler<&'cx muilib::AppResources, ()> for App<'cx> {
    fn new(resources: &'cx muilib::AppResources, event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Software Rasterizer"))
                .unwrap(),
        );

        let (uicx, canvas) =
            muilib::UiContext::create_for_window(resources, window.clone()).unwrap();
        Self {
            canvas,
            overlay_text_view: muilib::TextView::new(&uicx)
                .with_font_size(16.)
                .with_bg_color(Srgba::from_hex(0x80808080)),
            image_view: muilib::ImageView::new(RectSize::new(480., 320.)),
            game: softras::Game::new(),
            uicx,
            window,
            cursor_captured: false,
            cursor_position: None,
        }
    }
}

impl<'cx> ApplicationHandler for App<'cx> {
    fn resumed(&mut self, _: &ActiveEventLoop) {
        if self.cursor_captured {
            self.capture_cursor();
        } else {
            self.uncapture_cursor();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                let canvas_ref = self.canvas.create_ref().unwrap();
                self.draw_frame(canvas_ref);
                self.window.pre_present_notify();
                self.canvas.finish_drawing().unwrap();
                self.window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                self.canvas.reconfigure_for_size(
                    self.uicx.wgpu_device(),
                    size,
                    self.window.scale_factor(),
                    None,
                );
                let width_f = size.width as f32;
                let height_f = size.height as f32;
                let preferred_width: f32 = match cfg!(debug_assertions) {
                    true => 240.,
                    false => 800.,
                };
                let preferred_height = 0.75 * preferred_width;
                let (width, height) = match width_f > height_f {
                    true => (preferred_width, preferred_width / width_f * height_f),
                    false => (preferred_height / height_f * width_f, preferred_height),
                };
                self.game.notify_display_resize(width as u32, height as u32);
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: false,
            } => {
                let key_code = match event.physical_key {
                    PhysicalKey::Code(key_code_winit) => key_code::winit_to_softras(key_code_winit),
                    PhysicalKey::Unidentified(_) => softras::KeyCode::Unidentified,
                };
                if event.state.is_pressed() {
                    self.game.notify_key_down(key_code);
                } else {
                    self.game.notify_key_up(key_code);
                }
            }
            WindowEvent::CursorEntered { device_id: _ } => {
                self.game.notify_cursor_entered();
            }
            WindowEvent::CursorLeft { device_id: _ } => {
                self.game.notify_cursor_left();
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                self.cursor_position = Some(vec2(position.x as f32, position.y as f32));
            }
            WindowEvent::MouseWheel {
                device_id: _,
                delta: MouseScrollDelta::LineDelta(x, y),
                phase: _,
            } => {
                self.game.notify_cursor_scrolled_lines(x, y);
            }
            WindowEvent::MouseWheel {
                device_id: _,
                delta: MouseScrollDelta::PixelDelta(delta),
                phase: _,
            } => {
                self.game
                    .notify_cursor_scrolled_pixels(delta.x as f32, delta.y as f32);
            }
            WindowEvent::Focused(true) => {
                self.game.notify_focused();
                self.window.request_redraw();
            }
            WindowEvent::Focused(false) => {
                self.cursor_position = None;
                self.game.notify_unfocused();
            }
            _ => (),
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && let Some(cursor_position) = self.cursor_position
        {
            self.game.notify_cursor_moved(
                cursor_position.x,
                cursor_position.y,
                delta.0 as f32,
                delta.1 as f32,
            );
        }
    }
}
