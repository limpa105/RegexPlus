// This is the custom jsPsych plugin for our experiment.
// It is based on the jsPsych HTML survey plugin, which is MIT licensed.

import { JsPsych, JsPsychPlugin, ParameterType, TrialType } from "jspsych";

export type ListenerData = {
  examples: string,
  regex: string,
};

const info = <const>{
  name: "regex-examples",
  parameters: {
    examples: {
      type: ParameterType.STRING,
      pretty_name: "Examples",
      default: "Oh no someone forgot to provide examples :(",
    },

    regex: {
      type: ParameterType.STRING,
      pretty_name: "Regex",
      default: "",
    },

    NUM_TASKS: {
      type: ParameterType.INT,
      pretty_name: "Number of Tasks",
      default: 1,
    }
  },
};

type Info = typeof info;

export class RegexExamplesPlugin implements JsPsychPlugin<Info> {
  static info = info;

  constructor(private jsPsych: JsPsych) {}

  trial(display_element: HTMLElement, trial: TrialType<Info>) {
    console.log(`Asking for examples for <tt> ${trial.regex} </tt>`);
    var html = "";
    html += `
    <div>Guess the regular expression described using the examples below:</div>
    <p><<span class=examples>${trial.examples}<span></p>
    `;
    // start form
    html += `<form id="regex-examples-form" autocomplete="off">
    <!-- Prevent implicit submission of the form; see https://stackoverflow.com/a/51507806 -->
    <button type="submit" disabled style="display: none" aria-hidden="true"></button>
    `;
    // add guess button
    html +=
      '<input type="Guess" id="jspsych-survey-html-form-next" class="jspsych-btn jspsych-survey-html-form" value="Next> "></input>';

    html += "</form>";
    display_element.innerHTML = html;

    const startTime = performance.now();

    //const add_button = display_element.querySelector("#add");
    //add_button.addEventListener("click", () => AddEx(trial.regex, (add_button as any)));

    // Add the first example box
    //AddEx(trial.regex, add_button as any);

    const this_form = display_element.querySelector("#regex-examples-form");
    this_form.addEventListener("submit", event => {
      // don't submit form
      event.preventDefault();

      if (document.getElementsByClassName("wrong").length > 0) {
        alert("Not all your examples are valid: please make sure they match the description");
        return;
      }
      // Commenting this out for now to allow 
      //if (document.getElementsByClassName("correct").length < 1) {
        //alert("Please add at least one example!");
        //return;
      //}

      // measure response time
      const endTime = performance.now();
      const response_time = Math.round(endTime - startTime);

      const question_data = getArrayOfExamples(this_form);

      // save data
      const trialdata = {
        response_time: response_time,
        regex: trial.regex,
        response: question_data,
      };

      display_element.innerHTML = "";

      // next trial
      const progress = this.jsPsych.getProgressBarCompleted();
      this.jsPsych.setProgressBar(progress + 1/trial.NUM_TASKS)
      this.jsPsych.finishTrial(trialdata);
    });
  }
}

/** Make an array of all the examples */
function getArrayOfExamples(form: any) {
  return [...form.elements]
    .filter(field => field.type === "text")
    .map(field => field.value);
}


/*********** Handlers for various interation events ************/

// Our functions :) 
async function RemoveEx(e : HTMLElement) {
  const div = e.parentNode;
  (div as any).remove();
}

async function AddEx(regex: string, e: HTMLElement) {
  const main = document.createElement("div")
  const br = document.createElement("br");
  main.appendChild(br)
  const button = document.createElement("button")
  button.type = "button";
  button.innerHTML = "Remove";
  main.appendChild(button)
  button.addEventListener('click', e => RemoveEx(e.target as any))
  const ex = document.createElement("span")
  ex.innerHTML = " Example: "
  main.appendChild(ex)
  const input = document.createElement("input")
  input.type = "text"
  input.addEventListener('keyup', e => updateWhetherItIsValid(regex, e.target))
  const correct = document.createElement("span")
  correct.innerHTML = " Invalid"
  correct.className = "wrong"
  main.appendChild(input)
  main.appendChild(correct)
  updateWhetherItIsValid(regex, input)
  document.getElementById("the_examples").appendChild(main)
}

function updateWhetherItIsValid(regex: string, t: any) {
  const re = new RegExp("^" + regex + "$");

  if ( re.test(t.value)) {
    const parent = t.parentNode
    const feedback = parent.getElementsByTagName("span")[1]
    feedback.innerHTML = " Valid"
    feedback.className  = "correct"
  } else {
    const parent = t.parentNode
    const feedback = parent.getElementsByTagName("span")[1]
    feedback.innerHTML = " Invalid"
    feedback.className  = "wrong"
  }
}


/*
// 1. Parsing
// 2. Brzozowski derivatives
// 3. Profit

const ALLOWED_CHAR = ['[0-9]', '[a-z]', '[A-Z]', '[a-zA-Z]', '[a-zA-Z0-9]']

type Atom =
  | { tag: 'const', data: string }
  | { tag: 'cc', data: Set<string>, plus: boolean }
  | { tag: 'opt', data: Atom }

type DSL = Atom[]

/// Returns a string describing the parse error if it does not parse
function parse_regex(input: string) : string | DSL {
  // regex ::= (simple_atom modifier*)*
  // modifier ::= ? or * or +
  // simple_atom ::= character class or text
  function parse_atom(input: string) : "error" | [Atom, string] {
    // check if character class 
     if (input[0] === '[') {
        if (input.indexOf(']') === -1){
            return 'error'
        }
        let char_class = input.substring(0, input.indexOf(']')+1)
        if (!ALLOWED_CHAR.includes(char_class)){
            return 'error'
        }
        


        // check if in permitted optionals

     }
    // check if optional 
    else if (input[0] === '(') {
        
    // will have to parse atom and then combine into one atom if its two or more 
        
    }
    else {
        let new_atom: Atom = {tag: 'const', data: input[0] }
        return [new_atom, input.substring(1)]
    }
    

    // try to parse an atom from the start of the input.
    // return the [atom, rest of input]
  }
  function parse_modifier(input: string, atom: Atom) : "error" | "no more modifiers" | [DSL, string] {
    // try to parse a modifier and apply it to the atom
    // modifiers: +, *, ?, {n}
    if (input === "") return "no more modifiers";
    if (input[0] === "+") {
        if (atom.tag !== 'cc')
            return "error"
        atom.plus = true;
        return [[atom], input.substring(1)];
    } else if (input[0] == "*") {
        if (atom.tag !== 'cc'){
            return "error"
        }
        atom.plus = true 
        let new_atom:Atom = {tag: 'opt', data: atom }
        return [[new_atom], input.substring(1)]
        // make it *
    } else if (input[0] === "?") {
        if (atom.tag ==='opt'){
            return 'error'
        }
        let new_atom:Atom = {tag: 'opt', data: atom }
        return [[new_atom], input.substring(1)]
        
        // make it optional
    } else if (input[0] === "{") {
        //#{n,m}
        let n = parseInt(input.substring(1))
        if (atom.tag !=='cc'){
            return 'error'
        }
        let mini_dsl: DSL = []
        for( let i=0; i<n; i++){
            mini_dsl.push({tag: 'cc', data: atom.data, plus: false})
        }
        // check if we are in {n,m} case?
        if (input.substring[n.toString().length] === ',') {
            let opt_num = parseInt(input.substring(n.toString().length +1)) - n 
            for (let i = 0; i<opt_num; i++){
                mini_dsl.push({tag: 'opt', data: atom})
            }

        }
        return [mini_dsl, input.substring(input.indexOf('}')+1)]   
    } else {
        return "no more modifiers";
    }
  }

  const result: DSL = [];
  while (input !== "") {
    const x = parse_atom(input);
    if (x === "error") return "parse error";
    let dsl = [x[0]];
    input = x[1];
    while (dsl.length === 1) {
      const x = parse_modifier(input, dsl[0]);
      if (x === "error") return "parse error";
      if (x === "no more modifiers") break;
      dsl = x[0];
      input = x[1];
    } 
    result.push(...dsl);
  }
  return result;
}

function matches(re: DSL, example: string) : boolean {
  // uses legal_chars and step
  // TODO
}

function dsl_same_grammar(a: DSL, b: DSL) : boolean {
  // uses legal_chars and step and dsl_syntax_equals and implements bisimilarity checking
  // TODO
}

function atom_syntax_equals(a: Atom, b: Atom) : boolean {
  switch (a.tag + "," + b.tag) {
    case "const,const": return a.data === b.data;
    case "cc,cc":
      return a.data.size === b.data.size \
        && a.data.forEach(x => b.data.has(x)) \
        && a.plus === b.plus;
    case "opt,opt": return atom_syntax_equals(a.data, b.data);
    default: return false;
  }
}
function dsl_syntax_equals(a: DSL, b: DSL) : boolean {
  if (a.length !== b.length) return false;
  return a.every((x,i) => atom_syntax_equals(x, b[i]));
}

type State = DSL[]

function normalize(st: State) : State {
  const res: State = [];
  st.forEach(x => {
    if (x.length !== 0 && out.every(y => !dsl_syntax_equals(x, y)))
      res.push(x);
  });
  return res;
}
function legal_chars(st: State) : Set<string> {
  const res: Set<string> = new Set();
  st.forEach(dsl => {
    let i = 0;
    do {
      let atom = dsl[i];
      const was_opt = atom.tag === "opt";
      while (atom.tag === "opt") atom = atom.data;
      switch (a.tag) {
        case "const":
          if (a.data.length !== 0) res.add(a.data[0]);
          break;
        case "cc":
          a.data.forEach(x => res.add(x));
          break;
      }
      i++;
    } while (was_opt);
  })
}
function step(st: State, char: string) : State {
  // this is Brzozowski derivative more or less
  const res: State = [];
  st.forEach(dsl => {
    if (dsl[0].tag === )
  });
  return normalize(res);
}

*/