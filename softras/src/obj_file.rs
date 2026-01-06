use std::{num::NonZeroU32, str};

use crate::*;

pub fn load_obj(data: &str) -> Vec<Vertex> {
    let mut state = LoaderState::default();
    for line in data.lines() {
        state.read_line(line);
    }
    state.finish()
}

#[derive(Default, Debug, Clone)]
struct LoaderState {
    v: Vec<Vec3>,
    vt: Vec<Vec2>,
    vn: Vec<Vec3>,
    faces: Vec<[FaceElement; 3]>,
}

impl LoaderState {
    fn read_line(&mut self, line: &str) {
        let mut words = line.split_whitespace();
        let Some(first_word) = words.next() else {
            return;
        };
        match first_word {
            "v" => self.v.push(parse_v(words).unwrap_or_default()),
            "vt" => self.vt.push(parse_vt(words).unwrap_or_default()),
            "vn" => self.vn.push(parse_vn(words).unwrap_or_default()),
            "f" => {
                if let Some(face_element) = parse_f(words) {
                    self.faces.push(face_element)
                }
            }
            _ => (),
        }
    }

    fn finish(&mut self) -> Vec<Vertex> {
        let mut vertices = Vec::default();
        for &[face0, face1, face2] in &self.faces {
            let position0 = self.v.get(u32::from(face0.v) as usize - 1).copied();
            let position1 = self.v.get(u32::from(face1.v) as usize - 1).copied();
            let position2 = self.v.get(u32::from(face2.v) as usize - 1).copied();
            let (Some(p0), Some(p1), Some(p2)) = (position0, position1, position2) else {
                continue;
            };
            let normal = (p1 - p0).cross(p2 - p0).normalize();
            self.append_face(&mut vertices, face0, normal);
            self.append_face(&mut vertices, face1, normal);
            self.append_face(&mut vertices, face2, normal);
        }
        vertices
    }

    fn append_face(
        &self,
        vertices: &mut Vec<Vertex>,
        face_element: FaceElement,
        calculated_normal: Vec3,
    ) {
        let position = self
            .v
            .get(u32::from(face_element.v) as usize - 1)
            .copied()
            .unwrap_or_default();
        let uv = face_element
            .vt
            .and_then(|vt| self.vt.get(u32::from(vt) as usize - 1))
            .copied()
            .unwrap_or_default();
        let normal = face_element
            .vn
            .and_then(|vn| self.vn.get(u32::from(vn) as usize - 1))
            .copied()
            .unwrap_or(calculated_normal);
        vertices.push(Vertex {
            position,
            uv,
            normal,
        });
    }
}

fn parse_v(mut words: str::SplitWhitespace) -> Option<Vec3> {
    Some(vec3(
        words.next()?.parse().ok()?,
        words.next()?.parse().ok()?,
        words.next()?.parse().ok()?,
    ))
}

fn parse_vt(mut words: str::SplitWhitespace) -> Option<Vec2> {
    Some(vec2(
        words.next()?.parse().ok()?,
        words.next()?.parse().ok()?,
    ))
}

fn parse_vn(mut words: str::SplitWhitespace) -> Option<Vec3> {
    Some(vec3(
        words.next()?.parse().ok()?,
        words.next()?.parse().ok()?,
        words.next()?.parse().ok()?,
    ))
}

#[derive(Debug, Clone, Copy)]
struct FaceElement {
    v: NonZeroU32,
    vt: Option<NonZeroU32>,
    vn: Option<NonZeroU32>,
}

fn parse_f(mut words: str::SplitWhitespace) -> Option<[FaceElement; 3]> {
    Some([
        parse_face_element(words.next()?)?,
        parse_face_element(words.next()?)?,
        parse_face_element(words.next()?)?,
    ])
}

fn parse_face_element(s: &str) -> Option<FaceElement> {
    let first_char = s.chars().next()?;
    match first_char {
        'v' => todo!(),
        c if c.is_ascii_digit() => Some(FaceElement {
            v: s.parse().ok()?,
            vt: None,
            vn: None,
        }),
        _ => None,
    }
}
