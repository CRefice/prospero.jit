#[cfg(not(target_feature = "avx"))]
compile_error!("AVX is required for this project");

use prospero::Instr;
use prospero::codegen::{self, CodeBuffer, EntryPoint, Ymm};
use prospero::optimize::recursive_specialize;

use std::arch::x86_64::*;
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

fn to_image_bytes(x: Ymm) -> [u8; 8] {
    unsafe {
        let mask = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_setzero_ps());
        let mask: __m256i = std::mem::transmute(mask);
        let ones = _mm256_set1_epi32(255);
        let result = _mm256_and_si256(mask, ones);
        let result = _mm256_packus_epi32(result, result);
        let result = _mm256_packus_epi16(result, result);
        let result = _mm256_permutevar8x32_epi32(result, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_extract_epi64::<0>(result).to_le_bytes()
    }
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

    let timer = Instant::now();
    let specialized = recursive_specialize(instrs, num_splits);

    eprintln!("Compiled code in: {:?}", timer.elapsed());

    let offsets = unsafe {
        let offsets = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        let dividend = _mm256_set1_ps((image_size / 2) as f32);
        _mm256_div_ps(offsets, dividend)
    };

    struct Smuggle(*mut u8);
    unsafe impl Send for Smuggle {}
    impl Smuggle {
        fn as_slice(&mut self, len: usize) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.0, len) }
        }
    }

    let num_threads = std::thread::available_parallelism()
        .unwrap()
        .get()
        .min(num_splits);
    let block_size = image_size / num_splits;
    let blocks_per_thread = num_splits.div_ceil(num_threads);

    let mut image = vec![0u8; image_size * image_size];

    let timer = Instant::now();
    std::thread::scope(|s| {
        for thread in 0..num_threads {
            let mut image = Smuggle(image.as_mut_ptr());
            let chunk =
                &specialized[(thread * blocks_per_thread)..((thread + 1) * blocks_per_thread)];

            s.spawn(move || {
                let mut buf = CodeBuffer::default();
                let entrypoints: Vec<Vec<EntryPoint>> = chunk
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|instrs| {
                                let entrypoint = buf.entrypoint();
                                codegen::generate_code(&mut buf, instrs);
                                entrypoint
                            })
                            .collect()
                    })
                    .collect();
                let code = buf.install();
                let mut temp = code.allocate_temp_buf();

                for (row, y) in entrypoints.iter().zip((thread * blocks_per_thread)..) {
                    for (x, entry) in row.iter().enumerate() {
                        let start_y = y * block_size;
                        let end_y = start_y + block_size;
                        let start_x = x * block_size;
                        let end_x = start_x + block_size;
                        let image = image.as_slice(image_size * image_size);
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
                                    let result = code.invoke(*entry, x, y, &mut temp);
                                    chunk.copy_from_slice(&to_image_bytes(result));
                                }
                            }
                        }
                    }
                }
            });
        }
    });
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
