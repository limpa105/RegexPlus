use regex_plus::syn;

fn main() {
    println!("Inputs (leave blank when done):");
    let mut inputs = vec![];
    loop {
        let mut s = String::new();
        std::io::stdin().read_line(&mut s).unwrap();
        if s.trim() == "" {
            break;
        }
        inputs.push(s.trim().to_owned());
    }
    println!("VSA'ing");
    let best_regex = syn(&inputs);
    println!("Best regex: {}", best_regex);
}
