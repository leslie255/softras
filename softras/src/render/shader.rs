#![allow(dead_code)]

use bytemuck::{Pod, Zeroable};

use crate::render::*;

/// Analog of a shader in the renderer.
///
/// Obviously implemented CPU-side as oppose to GPU because this is a CPU-only renderer.
///
/// It is important to remember that, unlike a GPU shader, creation of a shader here has a
/// basically trivial cost. This difference might be a bit trippy sometimes for those who are used
/// to traditional GPU-based graphics programming.
pub trait FragmentShader {
    fn fragment(&self, input: FragmentInput) -> Rgba;
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct FragmentInput {
    pub position: Vec3,
    pub depth: f32,
    pub uv: Vec2,
    pub normal: Vec3,
}

/// Shader that just outputs a constant color.
/// For basic directional shading on 3D objects, use [`DirectionalShadingShader`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasicShader {
    /// The default value is `Rgb::from_hex(0xFF00FF)`.
    pub fill_color: Rgb,
}

impl Default for BasicShader {
    fn default() -> Self {
        Self {
            fill_color: Rgb::from_hex(0xFFFFFF),
        }
    }
}

impl FragmentShader for BasicShader {
    fn fragment(&self, _: FragmentInput) -> Rgba {
        self.fill_color.into()
    }
}

/// Fill color + normal-based directional shading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalShadingShader {
    /// The color of the object surface.
    ///
    /// The default value is `Rgb::from_hex(0xFFFFFF)`.
    pub color: Rgb,
    /// The direction of global directional light.
    ///
    /// The default value is `vec3(-1., -1., -1.).normalize()`.
    pub light_direction: Vec3,
    /// Controls the intensity of the shading effect.
    ///
    /// The default value is `0.5`.
    pub shading_intensity: f32,
    /// Controls how much of the lighting should be expressed in highlight vs. shadow.
    /// `highlightness = 0.` would cause only subtraction to the color values, i.e. shadow.
    /// `highlightness = 1.` would cause only addition to the color values, i.e. highlight.
    ///
    /// The default value is `0.6`.
    pub highlightness: f32,
}

impl Default for DirectionalShadingShader {
    fn default() -> Self {
        Self {
            color: Rgb::from_hex(0xFFFFFF),
            light_direction: vec3(-1., -1., -1.).normalize(),
            shading_intensity: 0.5,
            highlightness: 0.6,
        }
    }
}

impl FragmentShader for DirectionalShadingShader {
    fn fragment(&self, input: FragmentInput) -> Rgba {
        let normal = input.normal.normalize_or(vec3(1., 0., 0.));
        let light_direction = self.light_direction.normalize_or(vec3(1., 0., 0.));
        let theta = f32::acos(normal.dot(light_direction));
        let shading = theta / (2. * std::f32::consts::PI) - 1. + self.highlightness;
        Rgb {
            r: self.color.r + self.shading_intensity * shading,
            g: self.color.g + self.shading_intensity * shading,
            b: self.color.b + self.shading_intensity * shading,
        }
        .into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthVisualizationShader {
    pub z_min: f32,
    pub z_max: f32,
}

impl FragmentShader for DepthVisualizationShader {
    fn fragment(&self, input: FragmentInput) -> Rgba {
        match input.depth {
            depth if depth < 0. => Rgba::from_hex(0x00FFFFFF),
            depth if depth > 1. => Rgba::from_hex(0xFF0000FF),
            depth => {
                let result = (depth - self.z_min) / (self.z_max - self.z_min);
                Rgba {
                    r: result,
                    g: result,
                    b: result,
                    a: 1.,
                }
            }
        }
    }
}

pub trait PostprocessShader {
    fn postprocess(&self, input: PostprocessInput) -> Rgba;
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct PostprocessInput {
    pub albedo: Rgba,
    pub position: Vec3,
    pub depth: f32,
    pub normal: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasicPostprocessor {}

impl PostprocessShader for BasicPostprocessor {
    fn postprocess(&self, input: PostprocessInput) -> Rgba {
        input.albedo
    }
}
/// Fill color + normal-based directional shading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalShadingPostprocessor {
    /// The direction of global directional light.
    ///
    /// The default value is `vec3(-1., -1., -1.).normalize()`.
    pub light_direction: Vec3,
    /// Controls the intensity of the shading effect.
    ///
    /// The default value is `0.5`.
    pub shading_intensity: f32,
    /// Controls how much of the lighting should be expressed in highlight vs. shadow.
    /// `highlightness = 0.` would cause only subtraction to the color values, i.e. shadow.
    /// `highlightness = 1.` would cause only addition to the color values, i.e. highlight.
    ///
    /// The default value is `0.6`.
    pub highlightness: f32,
}

impl Default for DirectionalShadingPostprocessor {
    fn default() -> Self {
        Self {
            light_direction: vec3(-1., -1., -1.).normalize(),
            shading_intensity: 0.5,
            highlightness: 0.6,
        }
    }
}

impl PostprocessShader for DirectionalShadingPostprocessor {
    fn postprocess(&self, input: PostprocessInput) -> Rgba {
        let normal = input.normal.normalize_or(vec3(1., 0., 0.));
        let light_direction = self.light_direction.normalize_or(vec3(1., 0., 0.));
        let theta = f32::acos(normal.dot(light_direction));
        let shading = theta / (2. * std::f32::consts::PI) - 1. + self.highlightness;
        Rgb {
            r: input.albedo.r + self.shading_intensity * shading,
            g: input.albedo.g + self.shading_intensity * shading,
            b: input.albedo.b + self.shading_intensity * shading,
        }
        .into()
    }
}

