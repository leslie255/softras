#!/bin/zsh

set -xe

SOFTRAS_DEFAULT_ASSET_PATH="../Resources/assets.respack.bin" cargo build --quiet --release --package softras_winit_wgpu --target x86_64-apple-darwin
SOFTRAS_DEFAULT_ASSET_PATH="../Resources/assets.respack.bin" cargo build --quiet --release --package softras_winit_wgpu --target aarch64-apple-darwin

rm -rf "target/Software Rasterizer.app"
cp -r macos_bundle_template "target/Software Rasterizer.app"

lipo -create -output "target/Software Rasterizer.app/Contents/MacOS/softras_winit_wgpu" \
    "target/aarch64-apple-darwin/release/softras_winit_wgpu"                            \
    "target/x86_64-apple-darwin/release/softras_winit_wgpu"

cp "assets.respack.bin" "target/Software Rasterizer.app/Contents/Resources/assets.respack.bin"

