//! The actual search code

use crate::setup::*;
use typed_arena::Arena;

fn atom_to_string(interner: &Interner<'_>, atom: Atom, edge: Edge) -> String {
    let is_repeated = edge.flags & ONE_CHAR_BIT == 0;
    let is_opt = edge.flags & NON_OPTIONAL_BIT == 0;

    use Atom::*;
    let char_class = match atom {
        Digit => "\\d",
        Lower => "[a-z]",
        Upper => "[A-Z]",
        Alpha => "[A-Za-z]",
        Alnum => "[A-Za-z0-9]",
        Literal => {
            if is_opt {
                return format!("({})?", interner.str_of_id(edge.literal));
            } else {
                return interner.str_of_id(edge.literal).to_owned();
            }
        }
    };
    match (is_repeated, is_opt) {
        (false, false) => format!("{}", char_class),
        (false, true) => format!("{}?", char_class),
        (true, false) => format!("{}+", char_class),
        (true, true) => format!("{}*", char_class),
    }
}

fn process_edge(
    interner: &Interner<'_>,
    edge_cache: &EdgeCache,
    num_inputs: usize,
    edge: Edge,
    from_sum: usize,
    to_sum: usize,
) -> (Atom, f32) {
    let cost_of_atom = -0.95f32.ln();
    let is_repeated = edge.flags & ONE_CHAR_BIT == 0;
    let is_opt = edge.flags & NON_OPTIONAL_BIT == 0;
    let literal_cost = cost_of_atom
        + if edge.literal as usize > MAX_STR_ID {
            f32::INFINITY
        } else if is_opt {
            let simplicity = 10f32.ln() + interner.cost_of_id(edge.literal);
            let specificity = num_inputs as f32 * 2f32.ln();
            simplicity + specificity
        } else if is_repeated {
            0. // don't have repeated non-optional literals
        } else {
            95f32.ln() - 0.3f32.ln()
        };
    let (cc, cc_cost) = edge_cache[to_sum - from_sum][edge.flags as usize];
    if cc_cost < literal_cost {
        (cc, cc_cost)
    } else {
        (Atom::Literal, literal_cost)
    }
}

pub fn search(interner: &Interner<'_>, edge_cache: &EdgeCache, inputs: &[Input]) -> String {
    let arena = Arena::new();

    // Make a list of all the nodes
    let mut nodes: Vec<(usize, &[usize])> = vec![];
    foreach_product(inputs.iter().map(|i| 0..i.num_nodes()), |sum, node| {
        nodes.push((sum, arena.alloc_extend(node.iter().copied())))
    });
    nodes.sort_by_key(|&(sum, _)| sum);
    let mut indices = vec![0; nodes.len()];
    for (idx, (_, node)) in nodes.iter().enumerate() {
        let i = inputs
            .iter()
            .zip(node.iter())
            .fold(0, |acc, (inp, i)| acc * inp.num_nodes() + i);
        indices[i] = idx;
    }

    // Dynamic programming time: fill out the info tables, starting from the front
    println!("Dynamic programming time");
    let mut costs = vec![f32::INFINITY; nodes.len()];
    let mut atoms = vec![Atom::Literal; nodes.len()];
    let mut edges = vec![Edge::NOPE; nodes.len()];
    let mut preds = vec![800; nodes.len()];
    costs[0] = 0.;
    for to_idx in 1..nodes.len() {
        let (to_sum, to) = nodes[to_idx];
        let mut best_cost = f32::INFINITY;
        let mut best_atom = Atom::Literal;
        let mut best_edge = Edge::NOPE;
        let mut best_pred = 0;
        foreach_product(to.iter().map(|&i| 0..i + 1), |from_sum, from| {
            if from == to {
                return;
            }
            let i = inputs
                .iter()
                .zip(from.iter())
                .fold(0, |acc, (inp, i)| acc * inp.num_nodes() + i);
            let from_idx = indices[i];
            debug_assert_eq!(nodes[from_idx].1, from);

            let edge = inputs
                .iter()
                .zip(from.iter().zip(to.iter()))
                .map(|(input, (&a, &b))| input.edge(a, b))
                .reduce(Edge::intersect)
                .unwrap();

            let (atom, edge_wt) =
                process_edge(interner, edge_cache, inputs.len(), edge, from_sum, to_sum);
            let wt = costs[from_idx] + edge_wt;

            // Do a min-reduce
            if wt < best_cost {
                best_cost = wt;
                best_atom = atom;
                best_edge = edge;
                best_pred = from_idx;
            }
        });
        costs[to_idx] = best_cost;
        atoms[to_idx] = best_atom;
        edges[to_idx] = best_edge;
        preds[to_idx] = best_pred;
    }

    // Reconstruct the resulting regex
    // Really ought to do it recursively but that's a pain in Rust
    println!("Building regex");
    let mut result = String::new();
    let mut idx = nodes.len() - 1;
    while idx != 0 {
        result = atom_to_string(interner, atoms[idx], edges[idx]) + &result;
        idx = preds[idx];
    }
    return result;
}

/// Iterate thru the cartesian product of a buncha iterators. The function gets the list and its
/// sum
fn foreach_product<Iter, BigIter, F>(it: BigIter, mut f: F)
where
    BigIter: Clone + Iterator<Item = Iter>,
    Iter: Iterator<Item = usize>,
    F: for<'a> FnMut(usize, &'a [usize]),
{
    foreach_product_helper(&mut vec![], 0, it, &mut f)
}

fn foreach_product_helper<Iter, BigIter, F>(
    v: &mut Vec<usize>,
    acc: usize,
    mut it: BigIter,
    f: &mut F,
) where
    BigIter: Clone + Iterator<Item = Iter>,
    Iter: Iterator<Item = usize>,
    F: for<'a> FnMut(usize, &'a [usize]),
{
    if let Some(iter) = it.next() {
        for x in iter {
            v.push(x);
            foreach_product_helper(v, acc + x, it.clone(), f);
            v.pop();
        }
    } else {
        f(acc, &v[..]);
    }
}
