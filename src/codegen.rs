use std::alloc::{self, Layout};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy)]
pub enum Operand {
    Reg(u8),
    Memory { base: u8, disp: u32 },
}

impl Operand {
    fn register(&self) -> u8 {
        match *self {
            Self::Reg(x) => x,
            Self::Memory { base, .. } => base,
        }
    }
}

#[derive(Default)]
pub struct CodeBuffer {
    buf: Vec<u8>,
    stack_size: u32,
}

impl CodeBuffer {
    fn append(&mut self, byte: u8) {
        self.buf.push(byte);
    }

    fn vex_full(&mut self, reg: u8, vvvv: u8, r_m: u8, pp: u8, map: u8) {
        self.append(0xc4);

        {
            let r_bar = (!reg) & 0b1000;
            let x_bar = 1;
            let b_bar = (!r_m) & 0b1000;
            let mmmmm = map & 0b11111;
            self.append((r_bar << 4) | (x_bar << 6) | (b_bar << 2) | mmmmm);
        }

        {
            let w = 0;
            let vvvv_bar = (!vvvv) & 0b1111;
            let l = 1;
            let pp = pp & 0b11;
            self.append((w << 7) | (vvvv_bar << 3) | (l << 2) | pp);
        }
    }

    fn vex(&mut self, reg: u8, vvvv: u8, r_m: u8) {
        self.vex_full(reg, vvvv, r_m, 0, 1);
    }

    fn mod_r_m(&mut self, r#mod: u8, reg: u8, r_m: u8) {
        let reg = reg & 0b111;
        let r_m = r_m & 0b111;
        self.append((r#mod << 6) | (reg << 3) | r_m);
    }

    fn operands(&mut self, reg: u8, r_m: Operand) {
        match r_m {
            Operand::Reg(r_m) => self.mod_r_m(0b11, reg, r_m),
            Operand::Memory { base, disp } => {
                self.mod_r_m(0b10, reg, base);
                self.buf.extend_from_slice(&disp.to_le_bytes());
            }
        }
    }

    pub fn mov(&mut self, dest: Operand, source: Operand) {
        let (reg, r_m, opcode) = match (dest, source) {
            (Operand::Reg(reg), source) => (reg, source, 0x10),
            (Operand::Memory { .. }, Operand::Reg(reg)) => (reg, dest, 0x11),
            (x, y) => unreachable!("Unsupported mov: {:?} <- {:?}", x, y),
        };
        self.vex(reg, 0, r_m.register());
        self.append(opcode);
        self.operands(reg, r_m);
    }

    pub fn broadcast(&mut self, dest: u8, source: Operand) {
        let Operand::Memory { base, disp } = source else {
            unreachable!("Cannot broadcast register value: {:?}", source)
        };

        self.vex_full(dest, 0, base, 1, 2);
        self.append(0x18);
        self.operands(dest, Operand::Memory { base, disp });
    }

    pub fn add(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x58);
        self.operands(dest, y);
    }

    pub fn sub(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x5c);
        self.operands(dest, y);
    }

    pub fn mul(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x59);
        self.operands(dest, y);
    }

    pub fn max(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x5f);
        self.operands(dest, y);
    }

    pub fn min(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x5d);
        self.operands(dest, y);
    }

    pub fn xor(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x57);
        self.operands(dest, y);
    }

    pub fn sqrt(&mut self, dest: u8, val: Operand) {
        self.vex(dest, 0, val.register());
        self.append(0x51);
        self.operands(dest, val);
    }

    const SCRATCH: Operand = Operand::Reg(15);
    const CONSTANT_BASE: Operand = Operand::Reg(0);
    const TEMP_BASE: Operand = Operand::Reg(1);

    pub fn binary(&mut self, instr: &Instr, dest: u8, values: &[Operand]) {
        let (x, y) = instr.as_binary().expect("Not a binary instruction");
        let x = values[x.0 as usize];
        let y = values[y.0 as usize];
        let lhs = match x {
            Operand::Reg(reg) => reg,
            mem @ Operand::Memory { .. } => {
                self.mov(Self::SCRATCH, mem);
                Self::SCRATCH.register()
            }
        };
        match instr {
            Instr::Add(..) => self.add(dest, lhs, y),
            Instr::Sub(..) => self.sub(dest, lhs, y),
            Instr::Mul(..) => self.mul(dest, lhs, y),
            Instr::Max(..) => self.max(dest, lhs, y),
            Instr::Min(..) => self.min(dest, lhs, y),
            x => unreachable!("Not a binary instruction: {:?}", x),
        }
    }

    pub fn instruction(&mut self, instr: &Instr, dest: u8, values: &[Operand]) {
        match instr {
            Instr::Var(_) => (),
            Instr::Const(disp) => {
                self.broadcast(
                    dest,
                    Operand::Memory {
                        base: Self::CONSTANT_BASE.register(),
                        disp: *disp * std::mem::size_of::<f32>() as u32,
                    },
                );
            }
            Instr::Neg(x) => {
                let x = values[x.0 as usize];
                let scratch = Self::SCRATCH.register();
                self.xor(scratch, scratch, Self::SCRATCH);
                self.sub(dest, scratch, x);
            }
            Instr::Sqrt(x) => {
                let x = values[x.0 as usize];
                self.sqrt(dest, x);
            }
            _ => self.binary(instr, dest, values),
        }
    }

    const VALUE_SIZE: u32 = std::mem::size_of::<Ymm>() as u32;

    pub fn spill(&mut self, reg: u8) -> Operand {
        let stack_slot = Operand::Memory {
            base: CodeBuffer::TEMP_BASE.register(),
            disp: self.stack_size * Self::VALUE_SIZE,
        };
        self.stack_size += 1;

        self.mov(stack_slot, Operand::Reg(reg));
        stack_slot
    }

    pub fn install(self) -> InstalledCode {
        use libc::{_SC_PAGESIZE, sysconf};
        let page_size = unsafe { sysconf(_SC_PAGESIZE) } as usize;
        let num_pages = usize::max(1, self.buf.len().div_ceil(page_size));
        let layout =
            Layout::from_size_align(page_size * num_pages, page_size).expect("invalid layout");

        use std::arch::x86_64::_mm256_setzero_ps;
        let default: Ymm = unsafe { _mm256_setzero_ps() };
        let temp_buf = vec![default; self.stack_size as usize];
        unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            // Fill with RET instructions
            ptr.write_bytes(0xc3, layout.size());
            ptr.copy_from_nonoverlapping(self.buf.as_ptr(), self.buf.len());

            // Make memory executable and not writable
            libc::mprotect(
                ptr as *mut libc::c_void,
                layout.size(),
                libc::PROT_EXEC | libc::PROT_READ,
            );

            InstalledCode {
                buf: ptr,
                //
                code_size: self.buf.len(),
                layout,
                temp_buf,
            }
        }
    }
}

use crate::{Instr, OperandId};
pub fn generate_code(buf: &mut CodeBuffer, instrs: &[Instr]) {
    #[derive(Default, Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
    struct LiveInterval {
        // exclusive
        end: OperandId,
        // inclusive
        start: OperandId,
    }

    let ends = crate::compute_last_usage(instrs);

    let mut active: BTreeSet<LiveInterval> = BTreeSet::new();

    // ymm0 and ymm1 are occupied by params
    // keep ymm15 as scratch register for spilled values
    let mut available_regs: Vec<u8> = (2..15).rev().collect();

    let mut locations: Vec<Operand> = Vec::new();

    let Some((last, instrs)) = instrs.split_last() else {
        // No need to do anything if there are no instructions
        return;
    };
    for (i, instr) in instrs.iter().enumerate() {
        let id = OperandId(i as u32);
        let end = ends[i];

        // Make unused registers available
        while let Some(i) = active.first().copied() {
            if i.end > id {
                break;
            }
            active.pop_first();
            let Operand::Reg(reg) = locations[i.start.0 as usize] else {
                unreachable!("active value location can only be register: {:?}", i);
            };
            available_regs.push(reg);
        }

        if let Instr::Var(reg) = instr {
            locations.push(Operand::Reg(*reg as u8));
            // No need to generate anything, arguments are already stored in registers
            // at the beginning of the function
            active.insert(LiveInterval { end, start: id });
        } else if let Some(reg) = available_regs.pop() {
            buf.instruction(instr, reg, &locations);
            locations.push(Operand::Reg(reg));
            active.insert(LiveInterval { end, start: id });
        } else {
            let candidate = active.last().expect("There's no live value topispill");
            let candidate_id = candidate.start.0 as usize;
            if candidate.end > end {
                let loc @ Operand::Reg(reg) = locations[candidate_id] else {
                    unreachable!("Cannot spill from memory");
                };

                locations[candidate_id] = buf.spill(reg);
                active.pop_last();

                buf.instruction(instr, reg, &locations);
                locations.push(loc);
                active.insert(LiveInterval { end, start: id });
            } else {
                let scratch = CodeBuffer::SCRATCH.register();
                buf.instruction(instr, scratch, &locations);
                locations.push(buf.spill(scratch));
            }
        }
    }
    // Last element is the returned value, so it must go in ymm0
    buf.instruction(last, 0, &locations);

    eprintln!("Spilled {} variables to memory", buf.stack_size);
}

pub struct InstalledCode {
    buf: *mut u8,
    code_size: usize,
    layout: Layout,
    temp_buf: Vec<Ymm>,
}

pub type Ymm = std::arch::x86_64::__m256;

impl InstalledCode {
    pub fn code(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buf, self.code_size) }
    }

    pub fn invoke(&self, x: Ymm, y: Ymm, constants: &[f32]) -> Ymm {
        type JitFn = extern "C" fn(Ymm, Ymm) -> Ymm;
        unsafe {
            let f: JitFn = std::mem::transmute(self.buf);
            let result: Ymm;
            std::arch::asm!(
                "call {}",
                in(reg) f,
                in("rax") constants.as_ptr(),
                in("rcx") self.temp_buf.as_ptr(),
                inout("ymm0") x => result,
                in("ymm1") y,
                clobber_abi("C"),
                options(nostack),
            );
            result
        }
    }
}
