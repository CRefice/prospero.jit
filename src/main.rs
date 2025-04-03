#[cfg(not(target_feature = "avx"))]
compile_error!("AVX is required for this project");

mod codegen;

use codegen::{CodeBuffer, Ymm};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;
use std::time::Instant;

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy, Default)]
pub struct VarId(u32);

impl VarId {
    fn parse(label: &str) -> Self {
        debug_assert!(label.starts_with('_'), "Must start with _: {}", label);
        let id = u32::from_str_radix(&label[1..], 16).expect("Could not parse label");
        VarId(id)
    }
}

#[derive(Debug, Clone, Copy)]
enum UnaryOpcode {
    Neg,
    Sqrt,
}

#[derive(Debug, Clone, Copy)]
enum BinaryOpcode {
    Add,
    Sub,
    Mul,
    Max,
    Min,
}

#[derive(Clone)]
enum Instr {
    Var(u32),
    Const(u32),
    Unary {
        op: UnaryOpcode,
        operand: VarId,
    },
    Binary {
        op: BinaryOpcode,
        lhs: VarId,
        rhs: VarId,
    },
}

impl std::fmt::Debug for Instr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instr::Var(i) => write!(f, "var-{}", i),
            Instr::Const(_) => write!(f, "const"),
            Instr::Unary { op, operand } => {
                let op = match op {
                    UnaryOpcode::Neg => "neg",
                    UnaryOpcode::Sqrt => "sqrt",
                };
                write!(f, "{} {}", op, operand.0)
            }
            Instr::Binary { op, lhs, rhs } => {
                let op = match op {
                    BinaryOpcode::Mul if lhs == rhs => {
                        return write!(f, "square {}", lhs.0);
                    }
                    BinaryOpcode::Add => "add",
                    BinaryOpcode::Sub => "sub",
                    BinaryOpcode::Mul => "mul",
                    BinaryOpcode::Max => "max",
                    BinaryOpcode::Min => "min",
                };
                write!(f, "{} {} {}", op, lhs.0, rhs.0)
            }
        }
    }
}

#[derive(Default)]
struct Parser {
    constants: Vec<f32>,
    param_count: u32,
}

impl Parser {
    fn parse<'a>(&mut self, mut it: impl Iterator<Item = &'a str>) -> Instr {
        match it.next().expect("Opcode must be present") {
            "const" => {
                let cnst = it.next().expect("Constant value must be present");
                let cnst = cnst
                    .parse::<f32>()
                    .expect("Could not parse f32 from string");

                let instr = Instr::Const(self.constants.len() as u32);
                self.constants.push(cnst);
                instr
            }
            x if x.starts_with("var") => {
                let instr = Instr::Var(self.param_count);
                self.param_count += 1;
                instr
            }
            "neg" => {
                let operand = VarId::parse(it.next().expect("Operand must be present"));
                Instr::Unary {
                    op: UnaryOpcode::Neg,
                    operand,
                }
            }
            "square" => {
                let operand = VarId::parse(it.next().expect("Operand must be present"));
                Instr::Binary {
                    op: BinaryOpcode::Mul,
                    lhs: operand,
                    rhs: operand,
                }
            }
            "sqrt" => {
                let operand = VarId::parse(it.next().expect("Operand must be present"));
                Instr::Unary {
                    op: UnaryOpcode::Sqrt,
                    operand,
                }
            }
            x => {
                let lhs = VarId::parse(it.next().expect("Left operand must be present"));
                let rhs = VarId::parse(it.next().expect("Right operand must be present"));
                use BinaryOpcode::*;
                let op = match x {
                    "add" => Add,
                    "sub" => Sub,
                    "mul" => Mul,
                    "max" => Max,
                    "min" => Min,
                    x => unreachable!("Unexpected opcode: {}", x),
                };
                Instr::Binary { op, lhs, rhs }
            }
        }
    }
}

impl Instr {
    fn traverse_inputs(&self, mut f: impl FnMut(VarId)) {
        match self {
            Instr::Binary { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            }
            Instr::Unary { operand, .. } => {
                f(*operand);
            }
            _ => (),
        }
    }
}

fn compute_last_usage(instrs: &[Instr]) -> Vec<VarId> {
    let mut uses: Vec<VarId> = Vec::new();
    uses.resize_with(instrs.len(), Default::default);
    for (id, i) in instrs.iter().enumerate() {
        let id = VarId(id as u32);
        i.traverse_inputs(|input| uses[input.0 as usize] = id);
    }
    uses
}

#[derive(Debug, Clone, Copy)]
struct Range {
    min: f32,
    max: f32,
}

fn range_for(elems: &[f32]) -> Range {
    let min = elems.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = elems.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    Range { min, max }
}

fn range_optimization(
    param_ranges: [Range; 2],
    instrs: &[Instr],
    constants: &[f32],
) -> HashMap<VarId, VarId> {
    let mut ranges: Vec<Range> = Vec::new();
    let mut replacements: HashMap<VarId, VarId> = HashMap::new();
    for (i, instr) in instrs.iter().enumerate() {
        let id = VarId(i as u32);
        let range = match instr {
            Instr::Var(x) => param_ranges[*x as usize],
            Instr::Const(x) => {
                let c = constants[*x as usize];
                Range { min: c, max: c }
            }
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

fn specialize(mut instrs: Vec<Instr>, constants: &[f32], param_ranges: [Range; 2]) -> Vec<Instr> {
    let replacements = range_optimization(param_ranges, &instrs, constants);
    let old_last = VarId(instrs.len() as u32 - 1);
    if let Some(last) = replacements.get(&old_last) {
        instrs.truncate(last.0 as usize + 1)
    }
    apply_replacements(&mut instrs, &replacements);
    cleanup_unused(instrs)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("No argument provided");
    let image_size: usize = args
        .next()
        .map(|x| x.parse().expect("Could not parse image size"))
        .unwrap_or(2048);

    let num_splits: usize = args
        .next()
        .map(|x| x.parse().expect("Could not parse number of chunks"))
        .unwrap_or(16);

    let file = File::open(path).expect("Could not open input file");
    let file = BufReader::new(file);

    let timer = Instant::now();
    let mut parser = Parser::default();
    let instrs = file
        .lines()
        .map(|line| line.expect("Could not read line"))
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let mut parts = line.split_whitespace();
            let _label = parts.next().expect("Label must be present");
            parser.parse(parts)
        })
        .collect::<Vec<_>>();
    eprintln!("Parsed code in: {:?}", timer.elapsed());

    let constants = parser.constants;

    let ranges = (0..num_splits)
        .map(|i| {
            let size = 2.0 / num_splits as f32;
            let min = i as f32 * size - 1.0;
            let max = min + size;
            Range { min, max }
        })
        .collect::<Vec<_>>();

    let timer = Instant::now();
    let specialized: Vec<Vec<codegen::InstalledCode>> = ranges
        .iter()
        .rev()
        .map(|y| {
            ranges
                .iter()
                .map(|x| {
                    let instrs = specialize(instrs.clone(), &constants, [*x, *y]);
                    let mut buf = CodeBuffer::default();
                    codegen::generate_code(&mut buf, &instrs);
                    buf.install()
                })
                .collect()
        })
        .collect();
    eprintln!("Compiled code in: {:?}", timer.elapsed());

    fn to_unit_rect(i: usize, image_size: usize) -> f32 {
        let i = i as isize;
        let half_size = (image_size / 2) as isize;
        (i - half_size) as f32 / half_size as f32
    }

    use std::arch::x86_64::*;

    let offsets = unsafe {
        let offsets = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        let dividend = _mm256_set1_ps((image_size / 2) as f32);
        _mm256_div_ps(offsets, dividend)
    };

    fn to_image_bytes(x: Ymm) -> [u8; 8] {
        unsafe {
            let mask = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_setzero_ps());
            let ones = _mm256_set1_ps(255.0);
            let result = _mm256_and_ps(mask, ones);
            let result = _mm256_cvtps_epi32(result);
            let result = _mm256_packus_epi32(result, result);
            let result = _mm256_packus_epi16(result, result);
            let result =
                _mm256_permutevar8x32_epi32(result, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_extract_epi64::<0>(result).to_le_bytes()
        }
    }

    let timer = Instant::now();
    let mut image = vec![0u8; image_size * image_size];

    let block_size = image_size / num_splits;
    for (y, row) in specialized.into_iter().enumerate() {
        for (x, code) in row.into_iter().enumerate() {
            let start_y = y * block_size;
            let end_y = start_y + block_size;
            let start_x = x * block_size;
            let end_x = start_x + block_size;
            let mut temp = code.allocate_temp_buf();
            for y in start_y..end_y {
                let row = &mut image[image_size * y..];
                for x in (start_x..end_x).step_by(8) {
                    let chunk = &mut row[x..(x + 8)];
                    let y = to_unit_rect(image_size - y, image_size);
                    let x = to_unit_rect(x, image_size);
                    unsafe {
                        let y = _mm256_set1_ps(y);
                        let x = _mm256_set1_ps(x);
                        let x = _mm256_add_ps(x, offsets);
                        let result = code.invoke(x, y, &mut temp, &constants);
                        chunk.copy_from_slice(&to_image_bytes(result));
                    }
                }
            }
        }
    }
    eprintln!("Executed kernel in: {:?}", timer.elapsed());

    image::save_buffer(
        Path::new("image.png"),
        &image,
        image_size as u32,
        image_size as u32,
        image::ColorType::L8,
    )
    .expect("Could not save image");
    eprintln!("Saved image to image.png");
}
