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

    search(&interner, &inputs)
}
