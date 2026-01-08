#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepRangeInclusiveU32 {
    pub start: u32,
    pub end: u32,
    pub step: u32,
}

impl Iterator for StepRangeInclusiveU32 {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start <= self.end {
            let v = self.start;
            self.start = v + self.step;
            Some(v)
        } else {
            None
        }
    }
}

pub fn step_range(start: u32, end: u32, step: u32) -> StepRangeInclusiveU32 {
    StepRangeInclusiveU32 { start, end, step }
}
