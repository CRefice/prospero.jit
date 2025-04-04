use crate::{BinaryOpcode, Instr, UnaryOpcode, VarId};

use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub min: f32,
    pub max: f32,
}

fn range_for(elems: &[f32]) -> Range {
    let min = elems.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = elems.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    Range { min, max }
}

fn range_optimization(param_ranges: &[Range], instrs: &[Instr]) -> HashMap<VarId, VarId> {
    let mut ranges: Vec<Range> = Vec::new();
    let mut replacements: HashMap<VarId, VarId> = HashMap::new();
    for (i, instr) in instrs.iter().enumerate() {
        let id = VarId(i as u32);
        let range = match instr {
            Instr::Var(x) => param_ranges[*x as usize],
            Instr::Const(c) => Range { min: *c, max: *c },
            Instr::Unary { op, operand } => {
                let range = &ranges[operand.0 as usize];
                match op {
                    UnaryOpcode::Neg => Range {
                        max: -range.min,
                        min: -range.max,
                    },
                    UnaryOpcode::Sqrt => Range {
                        min: range.min.max(0.0).sqrt(),
                        max: range.max.sqrt(),
                    },
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
                    Mul if lhs == rhs => {
                        let min = if xr.min <= 0.0 && xr.max >= 0.0 {
                            0.0
                        } else {
                            f32::min(xr.min * xr.min, xr.max * xr.max)
                        };
                        let max = f32::max(xr.min * xr.min, xr.max * xr.max);
                        Range { min, max }
                    }
                    Mul => range_for(&[
                        xr.min * yr.min,
                        xr.min * yr.max,
                        xr.max * yr.min,
                        xr.max * yr.max,
                    ]),
                    Max => {
                        if xr.min > yr.max {
                            let lhs = replacements.get(lhs).copied().unwrap_or(*lhs);
                            replacements.insert(id, lhs);
                            *xr
                        } else if xr.max < yr.min {
                            let rhs = replacements.get(rhs).copied().unwrap_or(*rhs);
                            replacements.insert(id, rhs);
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
                            let rhs = replacements.get(rhs).copied().unwrap_or(*rhs);
                            replacements.insert(id, rhs);
                            *yr
                        } else if xr.max < yr.min {
                            let lhs = replacements.get(lhs).copied().unwrap_or(*lhs);
                            replacements.insert(id, lhs);
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
        ranges.push(range);
    }
    replacements
}

fn apply_replacements(instrs: &mut [Instr], replacements: &HashMap<VarId, VarId>) {
    for instr in instrs.iter_mut() {
        match instr {
            Instr::Unary { operand, .. } => {
                *operand = replacements.get(operand).copied().unwrap_or(*operand);
            }
            Instr::Binary { lhs, rhs, .. } => {
                *lhs = replacements.get(lhs).copied().unwrap_or(*lhs);
                *rhs = replacements.get(rhs).copied().unwrap_or(*rhs);
            }
            _ => (),
        }
    }
}

fn cleanup_unused(instrs: Vec<Instr>) -> Vec<Instr> {
    let mut is_used = vec![false; instrs.len()];
    *is_used.last_mut().unwrap() = true;
    for (i, instr) in instrs.iter().enumerate().rev() {
        if is_used[i] {
            instr.traverse_inputs(|x| is_used[x.0 as usize] = true);
        }
    }

    let mut ids = Vec::with_capacity(instrs.len());

    let mut retained = 0u32;
    instrs
        .into_iter()
        .zip(is_used)
        .filter_map(|(mut instr, is_used)| {
            ids.push(VarId(retained));
            if !is_used {
                return None;
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
            Some(instr)
        })
        .collect()
}

pub fn specialize(mut instrs: Vec<Instr>, param_ranges: &[Range]) -> Vec<Instr> {
    let replacements = range_optimization(param_ranges, &instrs);
    let old_last = VarId(instrs.len() as u32 - 1);
    if let Some(last) = replacements.get(&old_last) {
        instrs.truncate(last.0 as usize + 1)
    }
    apply_replacements(&mut instrs, &replacements);
    cleanup_unused(instrs)
}
