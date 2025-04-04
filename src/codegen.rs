use crate::{BinaryOpcode, Instr, UnaryOpcode, VarId};
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
    code: Vec<u8>,
    constants: Vec<f32>,
    stack_size: u32,
}

impl CodeBuffer {
    fn append(&mut self, byte: u8) {
        self.code.push(byte);
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
                self.code.extend_from_slice(&disp.to_le_bytes());
            }
        }
    }

    fn mov(&mut self, dest: Operand, source: Operand) {
        let (reg, r_m, opcode) = match (dest, source) {
            (Operand::Reg(reg), source) => (reg, source, 0x10),
            (Operand::Memory { .. }, Operand::Reg(reg)) => (reg, dest, 0x11),
            (x, y) => unreachable!("Unsupported mov: {:?} <- {:?}", x, y),
        };
        self.vex(reg, 0, r_m.register());
        self.append(opcode);
        self.operands(reg, r_m);
    }

    fn broadcast(&mut self, dest: u8, source: Operand) {
        let Operand::Memory { base, disp } = source else {
            unreachable!("Cannot broadcast register value: {:?}", source)
        };

        self.vex_full(dest, 0, base, 1, 2);
        self.append(0x18);
        self.operands(dest, Operand::Memory { base, disp });
    }

    fn add(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x58);
        self.operands(dest, y);
    }

    fn sub(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x5c);
        self.operands(dest, y);
    }

    fn mul(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x59);
        self.operands(dest, y);
    }

    fn max(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x5f);
        self.operands(dest, y);
    }

    fn min(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x5d);
        self.operands(dest, y);
    }

    fn xor(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register());
        self.append(0x57);
        self.operands(dest, y);
    }

    fn sqrt(&mut self, dest: u8, val: Operand) {
        self.vex(dest, 0, val.register());
        self.append(0x51);
        self.operands(dest, val);
    }

    fn ret(&mut self) {
        self.append(0xc3);
    }

    fn constant(&mut self, cnst: f32) -> Operand {
        let slot = self.constants.len() as u32;
        self.constants.push(cnst);
        Operand::Memory {
            base: Self::RAX,
            disp: slot * std::mem::size_of::<f32>() as u32,
        }
    }

    const RAX: u8 = 0;
    const RCX: u8 = 1;
    const SCRATCH: u8 = 15;

    const VALUE_SIZE: u32 = std::mem::size_of::<Ymm>() as u32;

    pub fn install(self) -> InstalledCode {
        use libc::{_SC_PAGESIZE, sysconf};
        let page_size = unsafe { sysconf(_SC_PAGESIZE) } as usize;
        let num_pages = usize::max(1, self.code.len().div_ceil(page_size));
        let layout =
            Layout::from_size_align(page_size * num_pages, page_size).expect("invalid layout");

        unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            // Fill with RET instructions
            ptr.write_bytes(0xc3, layout.size());
            ptr.copy_from_nonoverlapping(self.code.as_ptr(), self.code.len());

            // Make memory executable and not writable
            libc::mprotect(
                ptr as *mut libc::c_void,
                layout.size(),
                libc::PROT_EXEC | libc::PROT_READ,
            );

            InstalledCode {
                code_buf: ptr,
                _code_size: self.code.len(),
                stack_size: self.stack_size as usize,
                constants: self.constants,
                layout,
            }
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
struct LiveInterval {
    // exclusive
    end: VarId,
    // inclusive
    start: VarId,
}

struct RegisterAllocator {
    assigned: Vec<Operand>,
    active: BTreeSet<LiveInterval>,
    available_regs: Vec<u8>,
    stack_size: u32,
}

impl std::ops::Index<VarId> for RegisterAllocator {
    type Output = Operand;

    fn index(&self, index: VarId) -> &Self::Output {
        &self.assigned[index.0 as usize]
    }
}

impl std::ops::IndexMut<VarId> for RegisterAllocator {
    fn index_mut(&mut self, index: VarId) -> &mut Self::Output {
        &mut self.assigned[index.0 as usize]
    }
}

impl RegisterAllocator {
    fn new(instrs: &[Instr]) -> Self {
        Self {
            assigned: Vec::with_capacity(instrs.len()),
            active: BTreeSet::new(),
            // ymm0 and ymm1 are occupied by params
            // keep ymm15 as scratch register for spilled values
            available_regs: (2..15).rev().collect(),
            stack_size: 0,
        }
    }

    fn free_dead_values(&mut self, cur: VarId) {
        // Make unused registers available
        while let Some(i) = self.active.first().copied() {
            if i.end > cur {
                break;
            }
            self.active.pop_first();
            match self[i.start] {
                Operand::Reg(reg) => self.available_regs.push(reg),
                _ => unreachable!(),
            }
        }
    }

    fn assign_register(&mut self, reg: u8, interval: LiveInterval) {
        self.assigned.push(Operand::Reg(reg));
        self.active.insert(interval);
    }

    fn new_stack_slot(&mut self) -> Operand {
        let disp = {
            let slot = self.stack_size;
            self.stack_size += 1;
            slot * CodeBuffer::VALUE_SIZE
        };
        Operand::Memory {
            base: CodeBuffer::RCX,
            disp,
        }
    }

    fn spill(&mut self, val: LiveInterval) -> (u8, Operand) {
        let Operand::Reg(reg) = self[val.start] else {
            panic!("Cannot spill a memory location: {:?}", self[val.start]);
        };

        let new_loc = self.new_stack_slot();
        self[val.start] = new_loc;
        (reg, new_loc)
    }

    fn binary(&mut self, buf: &mut CodeBuffer, instr: &Instr, dest: u8) {
        let Instr::Binary { op, lhs, rhs } = instr else {
            unreachable!("Not a binary instruction");
        };
        let (x, y) = (self[*lhs], self[*rhs]);
        let lhs = match x {
            Operand::Reg(reg) => reg,
            mem @ Operand::Memory { .. } => {
                buf.mov(Operand::Reg(CodeBuffer::SCRATCH), mem);
                CodeBuffer::SCRATCH
            }
        };
        match op {
            BinaryOpcode::Add => buf.add(dest, lhs, y),
            BinaryOpcode::Sub => buf.sub(dest, lhs, y),
            BinaryOpcode::Mul => buf.mul(dest, lhs, y),
            BinaryOpcode::Max => buf.max(dest, lhs, y),
            BinaryOpcode::Min => buf.min(dest, lhs, y),
        }
    }

    fn instruction(&mut self, buf: &mut CodeBuffer, instr: &Instr, dest: u8) {
        match instr {
            Instr::Var(_) => (),
            Instr::Const(cnst) => {
                let cnst = buf.constant(*cnst);
                buf.broadcast(dest, cnst);
            }
            Instr::Unary { op, operand } => {
                let x = self[*operand];
                match op {
                    UnaryOpcode::Neg => {
                        let scratch = CodeBuffer::SCRATCH;
                        buf.xor(scratch, scratch, Operand::Reg(scratch));
                        buf.sub(dest, scratch, x);
                    }
                    UnaryOpcode::Sqrt => buf.sqrt(dest, x),
                }
            }
            _ => self.binary(buf, instr, dest),
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

    fn generate_code(&mut self, buf: &mut CodeBuffer, instrs: &[Instr]) {
        let ends = Self::compute_last_usage(instrs);

        let Some((last, instrs)) = instrs.split_last() else {
            // No need to do anything if there are no instructions
            return;
        };

        for (i, instr) in instrs.iter().enumerate() {
            let interval = LiveInterval {
                end: ends[i],
                start: VarId(i as u32),
            };

            self.free_dead_values(interval.start);

            if let Instr::Var(reg) = instr {
                // No need to generate anything, arguments are already stored in registers
                // at the beginning of the function
                self.assign_register(*reg as u8, interval);
            } else if let Some(reg) = self.available_regs.pop() {
                self.instruction(buf, instr, reg);
                self.assign_register(reg, interval);
            } else {
                // Need to spill something

                let candidate = self
                    .active
                    .last()
                    .copied()
                    .expect("There's no live value to spill");

                if candidate.end > interval.end {
                    self.active.pop_last();

                    let (reg, mem) = self.spill(candidate);
                    buf.mov(mem, Operand::Reg(reg));
                    self.instruction(buf, instr, reg);
                    self.assign_register(reg, interval);
                } else {
                    let scratch = CodeBuffer::SCRATCH;
                    self.instruction(buf, instr, scratch);
                    let mem = self.new_stack_slot();
                    buf.mov(mem, Operand::Reg(scratch));
                    self.assigned.push(mem);
                }
            }
        }
        // Last element is the returned value, so it must go in ymm0
        self.instruction(buf, last, 0);
        buf.ret();

        buf.stack_size = self.stack_size;
    }
}

pub fn generate_code(buf: &mut CodeBuffer, instrs: &[Instr]) {
    RegisterAllocator::new(instrs).generate_code(buf, instrs);
}

pub struct InstalledCode {
    code_buf: *const u8,
    _code_size: usize,
    constants: Vec<f32>,
    stack_size: usize,
    layout: Layout,
}

impl Drop for InstalledCode {
    fn drop(&mut self) {
        use std::alloc;
        unsafe {
            libc::mprotect(
                self.code_buf as *mut libc::c_void,
                self.layout.size(),
                libc::PROT_READ | libc::PROT_WRITE,
            );
            alloc::dealloc(self.code_buf as *mut u8, self.layout);
        }
    }
}

unsafe impl Send for InstalledCode {}

impl InstalledCode {
    pub fn _code(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.code_buf, self._code_size) }
    }

    pub fn allocate_temp_buf(&self) -> Vec<Ymm> {
        let empty = unsafe { std::arch::x86_64::_mm256_setzero_ps() };
        vec![empty; self.stack_size]
    }
}

pub type Ymm = std::arch::x86_64::__m256;

impl InstalledCode {
    pub fn invoke(&self, x: Ymm, y: Ymm, temp: &mut [Ymm]) -> Ymm {
        unsafe {
            let fn_ptr = self.code_buf;
            let result: Ymm;
            std::arch::asm!(
                "call {}",
                in(reg) fn_ptr,
                in("rax") self.constants.as_ptr(),
                in("rcx") temp.as_mut_ptr(),
                inout("ymm0") x => result,
                inout("ymm1") y => _,
                out("ymm2") _,
                out("ymm3") _,
                out("ymm4") _,
                out("ymm5") _,
                out("ymm6")  _,
                out("ymm7")  _,
                out("ymm8")  _,
                out("ymm9")  _,
                out("ymm10") _,
                out("ymm11") _,
                out("ymm12") _,
                out("ymm13") _,
                out("ymm14") _,
                out("ymm15") _,
                options(nostack),
            );
            result
        }
    }
}
