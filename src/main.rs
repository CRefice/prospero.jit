#[cfg(not(target_feature = "avx"))]
compile_error!("AVX is required for this project");

mod codegen;

use codegen::{CodeBuffer, Ymm};
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;
use std::time::Instant;

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy, Default)]
pub struct OperandId(u32);

impl OperandId {
    fn parse(label: &str) -> Self {
        debug_assert!(label.starts_with('_'), "Must start with _: {}", label);
        let id = u32::from_str_radix(&label[1..], 16).expect("Could not parse label");
        OperandId(id)
    }
}

#[derive(Debug)]
pub enum Instr {
    Var(u32),
    Const(u32),
    Add(OperandId, OperandId),
    Sub(OperandId, OperandId),
    Mul(OperandId, OperandId),
    Max(OperandId, OperandId),
    Min(OperandId, OperandId),
    Neg(OperandId),
    Sqrt(OperandId),
}

#[derive(Default)]
pub struct Parser {
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
                let val = OperandId::parse(it.next().expect("Operand must be present"));
                Instr::Neg(val)
            }
            "square" => {
                let val = OperandId::parse(it.next().expect("Operand must be present"));
                Instr::Mul(val, val)
            }
            "sqrt" => {
                let val = OperandId::parse(it.next().expect("Operand must be present"));
                Instr::Sqrt(val)
            }
            x => {
                let l = OperandId::parse(it.next().expect("Left operand must be present"));
                let r = OperandId::parse(it.next().expect("Right operand must be present"));
                match x {
                    "add" => Instr::Add(l, r),
                    "sub" => Instr::Sub(l, r),
                    "mul" => Instr::Mul(l, r),
                    "max" => Instr::Max(l, r),
                    "min" => Instr::Min(l, r),
                    x => unreachable!("Unexpected opcode: {}", x),
                }
            }
        }
    }
}

impl Instr {
    fn traverse_inputs(&self, mut f: impl FnMut(OperandId)) {
        match self {
            Instr::Add(l, r)
            | Instr::Sub(l, r)
            | Instr::Mul(l, r)
            | Instr::Max(l, r)
            | Instr::Min(l, r) => {
                f(*l);
                f(*r);
            }
            Instr::Neg(x) | Instr::Sqrt(x) => {
                f(*x);
            }
            _ => (),
        }
    }

    fn as_binary(&self) -> Option<(OperandId, OperandId)> {
        match self {
            Instr::Add(l, r)
            | Instr::Sub(l, r)
            | Instr::Mul(l, r)
            | Instr::Max(l, r)
            | Instr::Min(l, r) => Some((*l, *r)),
            _ => None,
        }
    }
}

pub fn compute_last_usage(instrs: &[Instr]) -> Vec<OperandId> {
    let mut uses: Vec<OperandId> = Vec::new();
    uses.resize_with(instrs.len(), Default::default);
    for (id, i) in instrs.iter().enumerate() {
        let id = OperandId(id as u32);
        i.traverse_inputs(|input| uses[input.0 as usize] = id);
    }
    uses
}

fn main() {
    let path = std::env::args().nth(1).expect("No argument provided");
    let image_size = std::env::args()
        .nth(2)
        .expect("Image size required")
        .parse()
        .expect("Could not parse image size");

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

    let timer = Instant::now();
    let mut buf = CodeBuffer::default();
    codegen::generate_code(&mut buf, &instrs);
    eprintln!("Compiled code in: {:?}", timer.elapsed());

    let code = buf.install();

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
            let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_setzero_ps());
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
    for y in 0..image_size {
        let row = &mut image[image_size * y..];
        for x in (0..image_size).step_by(8) {
            let chunk = &mut row[x..(x + 8)];
            let y = to_unit_rect(image_size - y, image_size);
            let x = to_unit_rect(x, image_size);
            unsafe {
                let y = _mm256_set1_ps(y);
                let x = _mm256_set1_ps(x);
                let x = _mm256_add_ps(x, offsets);
                let result = code.invoke(x, y, &constants);
                chunk.copy_from_slice(&to_image_bytes(result));
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
