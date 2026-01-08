#![allow(dead_code)]

use bytemuck::{Pod, Zeroable};

use crate::render::*;

pub trait Material {
    fn fragment(&self, input: FragmentInput) -> FragmentOutput;
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct FragmentInput {
    pub position: Vec3,
    pub depth: f32,
    pub uv: Vec2,
    pub normal: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct FragmentOutput {
    pub albedo: Rgba,
    pub normal: Vec3,
}

pub mod materials {
    use crate::*;

    /// Material of just a constant color.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct ColorMaterial {
        /// The default value is `Rgb::from_hex(0xFF00FF)`.
        pub fill_color: Rgb,
    }

    impl Default for ColorMaterial {
        fn default() -> Self {
            Self {
                fill_color: Rgb::from_hex(0xFFFFFF),
            }
        }
    }

    impl Material for ColorMaterial {
        fn fragment(&self, input: FragmentInput) -> FragmentOutput {
            FragmentOutput {
                albedo: self.fill_color.into(),
                normal: input.normal,
            }
        }
    }
}

pub trait Postprocessor {
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

impl Postprocessor for BasicPostprocessor {
    fn postprocess(&self, input: PostprocessInput) -> Rgba {
        input.albedo
    }
}

pub mod postprocessors {
    use crate::*;

    /// Directly outputs color right out.
    #[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Basic;

    impl Postprocessor for Basic {
        fn postprocess(&self, input: PostprocessInput) -> Rgba {
            input.albedo
        }
    }

    /// Fill color + normal-based directional shading.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct DirectionalShading {
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

    impl Default for DirectionalShading {
        fn default() -> Self {
            Self {
                light_direction: vec3(-1., -1., -1.).normalize(),
                shading_intensity: 0.5,
                highlightness: 0.6,
            }
        }
    }

    impl Postprocessor for DirectionalShading {
        fn postprocess(&self, input: PostprocessInput) -> Rgba {
            if input.depth == 1. {
                return Rgba::from_hex(0);
            }
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
}
