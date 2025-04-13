use crate::{BinaryOpcode, Instr, UnaryOpcode, VarId};

#[derive(Default, Debug, Clone, Copy)]
pub struct Range {
    pub min: f32,
    pub max: f32,
}

fn range_for(elems: &[f32]) -> Range {
    let min = elems.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = elems.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    Range { min, max }
}

fn range_optimization(param_ranges: &[Range], instrs: &[Instr]) -> (Vec<VarId>, Range) {
    let mut ranges: Vec<Range> = vec![Default::default(); instrs.len()];
    let mut replacements: Vec<VarId> = (0..(instrs.len() as u32)).map(VarId).collect();
    for (i, instr) in instrs.iter().enumerate() {
        let range = match instr {
            Instr::Var(x) => param_ranges[*x as usize],
            Instr::Const(c) => Range { min: *c, max: *c },
            Instr::Unary { op, operand } => {
                let range = &ranges[operand.0 as usize];
                use UnaryOpcode::*;
                match op {
                    Neg => Range {
                        max: -range.min,
                        min: -range.max,
                    },
                    Sqrt => Range {
                        min: range.min.max(0.0).sqrt(),
                        max: range.max.sqrt(),
                    },
                    Square => {
                        let min = if range.min <= 0.0 && range.max >= 0.0 {
                            0.0
                        } else {
                            f32::min(range.min * range.min, range.max * range.max)
                        };
                        let max = f32::max(range.min * range.min, range.max * range.max);
                        Range { min, max }
                    }
                }
            }
            Instr::Binary { op, lhs, rhs } => {
                let xr = &ranges[lhs.0 as usize];
                let yr = &ranges[rhs.0 as usize];
                use BinaryOpcode::*;
                match op {
                    Add => Range {
                        min: xr.min + yr.min,
                        max: xr.max + yr.max,
                    },
                    Sub => Range {
                        min: xr.min - yr.max,
                        max: xr.max - yr.min,
                    },
                    Mul => range_for(&[
                        xr.min * yr.min,
                        xr.min * yr.max,
                        xr.max * yr.min,
                        xr.max * yr.max,
                    ]),
                    Max => {
                        if xr.min > yr.max {
                            let lhs = replacements[lhs.0 as usize];
                            replacements[i] = lhs;
                            *xr
                        } else if xr.max < yr.min {
                            let rhs = replacements[rhs.0 as usize];
                            replacements[i] = rhs;
                            *yr
                        } else {
                            Range {
                                min: xr.min.max(yr.min),
                                max: xr.max.max(yr.max),
                            }
                        }
                    }
                    Min => {
                        if xr.min > yr.max {
                            let rhs = replacements[rhs.0 as usize];
                            replacements[i] = rhs;
                            *yr
                        } else if xr.max < yr.min {
                            let lhs = replacements[lhs.0 as usize];
                            replacements[i] = lhs;
                            *xr
                        } else {
                            Range {
                                min: xr.min.min(yr.min),
                                max: xr.max.min(yr.max),
                            }
                        }
                    }
                }
            }
        };
        debug_assert!(range.min <= range.max, "{} <= {}", range.min, range.max);
        ranges[i] = range;
    }
    (replacements, *ranges.last().unwrap())
}

fn apply_replacements(instrs: &mut [Instr], replacements: &[VarId]) {
    for instr in instrs.iter_mut() {
        match instr {
            Instr::Unary { operand, .. } => {
                *operand = replacements[operand.0 as usize];
            }
            Instr::Binary { lhs, rhs, .. } => {
                *lhs = replacements[lhs.0 as usize];
                *rhs = replacements[rhs.0 as usize];
            }
            _ => (),
        }
    }
}

fn cleanup_unused(mut instrs: Vec<Instr>) -> Vec<Instr> {
    let mut is_used = vec![false; instrs.len()];
    *is_used.last_mut().unwrap() = true;
    for (i, instr) in instrs.iter().enumerate().rev() {
        if is_used[i] {
            instr.traverse_inputs(|x| is_used[x.0 as usize] = true);
        }
    }

    let mut ids = vec![VarId(0); instrs.len()];

    let mut retained = 0u32;
    let mut i = 0;
    instrs.retain_mut(|mut instr| {
        ids[i] = VarId(retained);
        if !is_used[i] {
            i += 1;
            return false;
        }

        match &mut instr {
            Instr::Unary { operand, .. } => {
                *operand = ids[operand.0 as usize];
            }
            Instr::Binary { lhs, rhs, .. } => {
                *lhs = ids[lhs.0 as usize];
                *rhs = ids[rhs.0 as usize];
            }
            _ => (),
        };
        retained += 1;
        i += 1;
        true
    });
    instrs
}

pub fn specialize(mut instrs: Vec<Instr>, param_ranges: &[Range]) -> Vec<Instr> {
    let (replacements, range) = range_optimization(param_ranges, &instrs);
    if range.min >= 0.0 {
        return vec![Instr::Const(1.0)];
    }
    if range.max < 0.0 {
        return vec![Instr::Const(-1.0)];
    }
    instrs.truncate(replacements.last().unwrap().0 as usize + 1);
    apply_replacements(&mut instrs, &replacements);
    cleanup_unused(instrs)
}

pub fn recursive_specialize(instrs: Vec<Instr>, num_splits: usize) -> Vec<Vec<Vec<Instr>>> {
    let range = Range {
        min: -1.0,
        max: 1.0,
    };
    let mut ret = vec![vec![Vec::new(); num_splits]; num_splits];
    recursive_specialize_mut(instrs, true, (0, 0), [range, range], num_splits, &mut ret);
    ret
}

pub fn recursive_specialize_mut(
    instrs: Vec<Instr>,
    horizontal_split: bool,
    (y, x): (usize, usize),
    ranges: [Range; 2],
    num_splits: usize,
    ret: &mut Vec<Vec<Vec<Instr>>>,
) {
    if num_splits == 1 {
        ret[y][x] = instrs;
        return;
    }

    if horizontal_split {
        let range = &ranges[1];
        let mid = range.min.midpoint(range.max);
        let bottom = Range {
            min: range.min,
            max: mid,
        };
        let bottom_ranges = [ranges[0], bottom];
        let top = Range {
            min: mid,
            max: range.max,
        };
        let top_ranges = [ranges[0], top];
        let bottom = specialize(instrs.clone(), &bottom_ranges);
        let top = specialize(instrs, &top_ranges);
        recursive_specialize_mut(
            bottom,
            !horizontal_split,
            (y * 2 + 1, x),
            bottom_ranges,
            num_splits,
            ret,
        );
        recursive_specialize_mut(
            top,
            !horizontal_split,
            (y * 2, x),
            top_ranges,
            num_splits,
            ret,
        );
    } else {
        let range = &ranges[0];
        let mid = range.min.midpoint(range.max);
        let left = Range {
            min: range.min,
            max: mid,
        };
        let left_ranges = [left, ranges[1]];
        let right = Range {
            min: mid,
            max: range.max,
        };
        let right_ranges = [right, ranges[1]];
        let left = specialize(instrs.clone(), &left_ranges);
        let right = specialize(instrs, &right_ranges);
        recursive_specialize_mut(
            left,
            !horizontal_split,
            (y, x * 2),
            left_ranges,
            num_splits / 2,
            ret,
        );
        recursive_specialize_mut(
            right,
            !horizontal_split,
            (y, x * 2 + 1),
            right_ranges,
            num_splits / 2,
            ret,
        );
    }
}
