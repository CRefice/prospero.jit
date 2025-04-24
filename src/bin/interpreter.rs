use prospero::Instr;

use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;
use std::time::Instant;

fn to_unit_rect(i: usize, image_size: usize) -> f32 {
    let i = i as isize;
    let half_size = (image_size / 2) as isize;
    (i - half_size) as f32 / half_size as f32
}

fn evaluate(instrs: &[Instr], x: f32, y: f32, temp: &mut [f32]) -> f32 {
    for (i, instr) in instrs.iter().enumerate() {
        let result = match *instr {
            Instr::Var(var) => {
                if var == 0 {
                    x
                } else {
                    y
                }
            }
            Instr::Const(x) => x,
            Instr::Unary { op, operand } => {
                let operand = temp[operand.0 as usize];
                match op {
                    prospero::UnaryOpcode::Neg => -operand,
                    prospero::UnaryOpcode::Square => operand * operand,
                    prospero::UnaryOpcode::Sqrt => operand.sqrt(),
                }
            }
            Instr::Binary { op, lhs, rhs } => {
                let lhs = temp[lhs.0 as usize];
                let rhs = temp[rhs.0 as usize];
                match op {
                    prospero::BinaryOpcode::Add => lhs + rhs,
                    prospero::BinaryOpcode::Sub => lhs - rhs,
                    prospero::BinaryOpcode::Mul => lhs * rhs,
                    prospero::BinaryOpcode::Max => lhs.max(rhs),
                    prospero::BinaryOpcode::Min => lhs.min(rhs),
                }
            }
        };
        temp[i] = result;
    }
    temp.last().copied().unwrap()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("No argument provided");
    let image_size: usize = args
        .next()
        .map(|x| x.parse().expect("Could not parse image size"))
        .unwrap_or(2048);

    let file = File::open(path).expect("Could not open input file");
    let file = BufReader::new(file);

    let timer = Instant::now();
    let instrs = file
        .lines()
        .map(|line| line.expect("Could not read line"))
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let mut parts = line.split_whitespace();
            let _label = parts.next().expect("Label must be present");
            Instr::parse(parts)
        })
        .collect::<Vec<_>>();
    eprintln!("Parsed code in: {:?}", timer.elapsed());

    let mut image = vec![0u8; image_size * image_size];
    let mut temp = vec![0f32; instrs.len()];

    let timer = Instant::now();
    for (y, row) in image.chunks_mut(image_size).enumerate() {
        for (x, pixel) in row.iter_mut().enumerate() {
            let x = to_unit_rect(x, image_size);
            let y = to_unit_rect(y, image_size);
            let val = evaluate(&instrs, x, y, &mut temp);
            *pixel = if val > 0.0 { 0 } else { 255 };
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
