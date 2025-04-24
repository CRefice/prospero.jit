pub mod codegen;
pub mod optimize;

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy, Default)]
pub struct VarId(pub u32);

impl VarId {
    pub fn parse(label: &str) -> Self {
        debug_assert!(label.starts_with('_'), "Must start with _: {}", label);
        let id = u32::from_str_radix(&label[1..], 16).expect("Could not parse label");
        VarId(id)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOpcode {
    Neg,
    Square,
    Sqrt,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOpcode {
    Add,
    Sub,
    Mul,
    Max,
    Min,
}

#[derive(Debug, Clone)]
pub enum Instr {
    Var(u32),
    Const(f32),
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

impl Instr {
    pub fn parse<'a>(mut it: impl Iterator<Item = &'a str>) -> Self {
        match it.next().expect("Opcode must be present") {
            "var-x" => Instr::Var(0),
            "var-y" => Instr::Var(1),
            "const" => {
                let cnst = it.next().expect("Constant value must be present");
                let cnst = cnst
                    .parse::<f32>()
                    .expect("Could not parse f32 from string");

                Instr::Const(cnst)
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
                Instr::Unary {
                    op: UnaryOpcode::Square,
                    operand,
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
                    x => panic!("Unexpected opcode: {}", x),
                };
                Instr::Binary { op, lhs, rhs }
            }
        }
    }

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
