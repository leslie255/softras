# SOFTRAS

**Software Rasterizer** (CPU-side 3D rendering)

<img width="600" alt="different objects" src="https://github.com/user-attachments/assets/4b31cc9b-3737-4358-9dfd-462883533150"/>

## Project Structure

The project is divided into two cargo crates:
- `softras_core`, where all the rendering logic happens, detached from any platform-specific APIs
- `softras_wgpu_winit`, the backend that manages window creation, frame presentation, and forwards UI events to the rendering module
- `softras_muilib`, another alternative backend based on my other project [`muilib`](https://github.com/leslie255/muilib)

## Building and Running

To build and run:

```bash
$ cargo run --release --package softras_winit_wgpu
```

This command has to be executed at the root of the project directory, as the program needs access to the `assets.respack.bin` file for game assets. Alternatively, you may provide the asset file path with the `--res` argument. For more information, see `cargo run --release --package softras_winit_wgpu -- --help`.

## Q: But why is there WGSL code?

The two backends `softras_muilib` and `softras_wgpu_winit`, which handle window creation and event handling, uses WGSL code in its resource directory to present the frame onto a window.

As stated in [Project Structure](https://github.com/leslie255/softras?tab=readme-ov-file#project-structure), the rendering happens in `softras_core`, and is platform-independent and does not access GPU.

## LICENSE

This project is licensed under Apache License 2.0.
