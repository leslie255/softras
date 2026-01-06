# SOFTRAS

**Software Rasterizer** (CPU-side 3D rendering)

<img width="400" alt="teapots" src="https://github.com/user-attachments/assets/2f4995e1-3099-488e-9bc6-bba9a6b5fb49"/> <img width="400" alt="Screenshot 2026-01-05 at 18 58 01" src="https://github.com/user-attachments/assets/c08e26e5-f2cf-4833-8ed0-bc9374ba7b7e"/>

## Project Structure

The project is divided into two cargo crates:
- `softras`, where all the rendering logic happens, detached from any platform-specific APIs
- `muilib_backend`, the backend that manages window creation and forwards UI events to the
  rendering module, it is based on my other project [`muilib`](https://github.com/leslie255/muilib),
  a GUI library powered by [`wgpu`](https://github.com/gfx-rs/wgpu)

## Building and Running

To build and run:

```bash
$ cargo run --release --package muilib_backend -- run
```

This command has to be executed at the root of the project directory, as `muilib_backend` reads from `muilib_backend/res` directory for resource files.

## Q: But why is there WGSL code?

`muilib_backend`, which handles window creation and event handling, uses WGSL code in its resource directory. This is because it uses the GUI library [`muilib`](https://github.com/leslie255/muilib) which requires some shader to be loaded to render GUI elements (the debug overlay text and presenting the frame buffer as an image, in the case of this project).

As stated in [Project Structure](https://github.com/leslie255/softras?tab=readme-ov-file#project-structure), the rendering happens in `softras`, and is platform-independent and does not access GPU.

## LICENSE

This project is licensed under Apache License 2.0.
