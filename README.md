# SOFTRAS

**Software Rasterizer**

CPU-side 3D rendering.
Designed to be generic over the backend so as to be extremely portable.

The project is divided into two cargo crates:
- `softras`, where all the rendering logic happens, detached from any platform-specific APIs
- `muilib_backend`, the backend that manages window creation and forwards UI events to the
  rendering module, it is based on my other project [`muilib`](https://github.com/leslie255/muilib),
  a GUI library powered by [`wgpu`](https://github.com/gfx-rs/wgpu)

## LICENSE

This project is licensed under Apache License 2.0.

