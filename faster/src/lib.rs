mod search;
mod setup;

use search::*;
use setup::*;

/// Synthesize a regex which matches these inputs
pub fn syn(inputs: &[String]) -> String {
    let mut interner = Interner::default();
    let inputs: Vec<Input> = inputs
        .iter()
        .map(|i| preprocess(i, &mut interner))
        .collect();

    let edge_cache = edge_cache(&inputs);

    search(&interner, &edge_cache, &inputs)
}
