//! All the preprocessing steps before the actual search

use std::collections::HashMap;
use std::iter;

pub type StrID = u32;

pub const ANY_STR: StrID = u32::MAX;
pub const NO_STR: StrID = u32::MAX - 1;
pub const MAX_STR_ID: usize = u32::MAX as usize - 2;

#[derive(Debug, Default)]
pub struct Interner<'a> {
    strings: Vec<&'a str>,
    costs: Vec<f32>,
    id_map: HashMap<&'a str, StrID>,
}

impl<'a> Interner<'a> {
    pub fn id_of_str(&mut self, s: &'a str) -> StrID {
        if let Some(&id) = self.id_map.get(s) {
            id
        } else {
            assert!(self.strings.len() < MAX_STR_ID);
            assert!(self.strings.len() == self.costs.len());
            let id = self.strings.len() as StrID;
            self.strings.push(s);
            self.costs.push(s.len() as f32 * 96f32.ln() + 95f32.ln());
            self.id_map.insert(s, id);
            id
        }
    }
    pub fn str_of_id(&self, id: StrID) -> &'a str {
        self.strings[id as usize]
    }
    pub fn cost_of_id(&self, id: StrID) -> f32 {
        if id == ANY_STR || id == NO_STR {
            f32::INFINITY
        } else {
            self.costs[id as usize]
        }
    }
}

/// 7 bits are used, five for character class and two for length.
///  - Does it match `\d*`
///  - Does it match `[a-z]*`
///  - Does it match `[A-Z]*`
///  - Does it match `[A-Za-z]*`
///  - Does it match `[A-Za-z0-9]*`
///  - Does it match `.?`
///  - Does it match `.+` (if not, it has to be an optional edge)
pub type Flags = u8;

pub const ANYTHING: Flags = 0b11_11111;
pub const NOTHING: Flags = 0b00_00000;

// Char class flags
pub const DIGIT_BIT: Flags = 0b00001;
pub const LOWER_BIT: Flags = 0b00010;
pub const UPPER_BIT: Flags = 0b00100;
pub const ALPHA_BIT: Flags = 0b01000;
pub const ALNUM_BIT: Flags = 0b10000;

pub const DIGIT: Flags = 0b10001;
pub const LOWER: Flags = 0b11010;
pub const UPPER: Flags = 0b11100;
pub const ALPHA: Flags = 0b11000;
pub const ALNUM: Flags = 0b10000;

pub const ANY_CHAR_CLASS: Flags = 0b11111;

// Length flags
pub const ONE_CHAR_BIT: Flags = 0b01_00000;
pub const NON_OPTIONAL_BIT: Flags = 0b10_00000;

pub const ONE_CHAR: Flags = 0b11_00000;
pub const MORE_THAN_ONE_CHAR: Flags = 0b10_00000;
pub const OPTIONAL: Flags = 0b01_00000;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    /// Info about char class and length. See [`Flags`]
    pub flags: Flags,
    /// - If `literal` is [`ANY_STR`], it is empty.
    /// - If `literal` is [`NO_STR`], it does not match `(skljsdfljk)?` for any constant string
    /// - If `literal` is `interner.str_to_id("skljsdfljk")`, it matches `(skljsdfljk)?`
    pub literal: StrID,
}
impl Edge {
    pub const NOPE: Self = Edge {
        flags: NOTHING,
        literal: NO_STR,
    };
    pub const EPSILON: Self = Edge {
        flags: OPTIONAL | ANY_CHAR_CLASS,
        literal: ANY_STR,
    };
    fn from_str<'a>(s: &'a str, interner: &mut Interner<'a>) -> Self {
        assert!(s.len() > 0);
        let classify_char = |c: char| {
            if c.is_ascii_digit() {
                DIGIT
            } else if c.is_ascii_lowercase() {
                LOWER
            } else if c.is_ascii_uppercase() {
                UPPER
            } else {
                0
            }
        };
        let char_class_flags = s.chars().fold(ANY_CHAR_CLASS, |x, c| x & classify_char(c));
        let length_flags = if s.chars().count() > 1 {
            MORE_THAN_ONE_CHAR
        } else {
            ONE_CHAR
        };
        Self {
            flags: char_class_flags | length_flags,
            literal: interner.id_of_str(s),
        }
    }

    pub fn intersect(self, other: Self) -> Self {
        Self {
            flags: self.flags & other.flags,
            literal: if self.literal == ANY_STR
                || other.literal == ANY_STR
                || self.literal == other.literal
            {
                // Relies on ANY_STR == 0xffffffff
                assert!(!ANY_STR == 0);
                self.literal & other.literal
            } else {
                NO_STR
            },
        }
    }
}

#[derive(Debug)]
pub struct Input {
    pub text: Vec<char>,
    edges: Vec<Edge>,
}

impl Input {
    pub fn num_chars(&self) -> usize {
        self.text.len()
    }
    pub fn num_nodes(&self) -> usize {
        self.text.len() + 1
    }
    pub fn edge(&self, i: usize, j: usize) -> Edge {
        let n = self.num_nodes();
        debug_assert!(i < n && j < n);
        self.edges[i * n + j]
    }
}

pub fn preprocess<'a>(input: &'a str, interner: &mut Interner<'a>) -> Input {
    let text: Vec<char> = input.chars().collect();
    let mut edges: Vec<Edge> = vec![];
    let char_boundaries = input
        .char_indices()
        .map(|(i, _)| i)
        .chain(iter::once(input.len()));
    for i in char_boundaries.clone() {
        for j in char_boundaries.clone() {
            let edge = if i > j {
                Edge::NOPE
            } else if i == j {
                Edge::EPSILON
            } else {
                Edge::from_str(&input[i..j], interner)
            };
            edges.push(edge);
        }
    }
    Input { text, edges }
}
