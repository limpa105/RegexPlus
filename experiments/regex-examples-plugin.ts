// This is the custom jsPsych plugin for our experiment.
// It is based on the jsPsych HTML survey plugin, which is MIT licensed.

import { JsPsych, JsPsychPlugin, ParameterType, TrialType } from "jspsych";

export type RegexExampleData = {
  description: string,
  regex: string,
};

const info = <const>{
  name: "regex-examples",
  parameters: {
    description: {
      type: ParameterType.STRING,
      pretty_name: "Description",
      default: "Oh no someone forgot to write a natural-language description for this one :'(",
    },

    regex: {
      type: ParameterType.STRING,
      pretty_name: "Regex",
      default: "",
    },
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
    <div>Please provide examples conveying the description below!</div>
    <p><strong>Description:</strong> <span class=description>${trial.description}<span></p>
    <p><strong>Corresponding Regex:</strong> <tt>  ${trial.regex} </tt> </p>
    `;
    // start form
    html += `<form id="regex-examples-form" autocomplete="off">
    <!-- Prevent implicit submission of the form; see https://stackoverflow.com/a/51507806 -->
    <button type="submit" disabled style="display: none" aria-hidden="true"></button>
    `;

    // add form HTML / input elements
    html += `<p id="the_examples"></p> <button id="add" class="jspsych-btn jspsych-survey-html-form" type="button"> I want to provide more examples! </button>`
    html += `<divclass="space">  </div> `
    // add submit button
    html +=
      '<input type="submit" id="jspsych-survey-html-form-next" class="jspsych-btn jspsych-survey-html-form" value="Next> "></input>';

    html += "</form>";
    display_element.innerHTML = html;

    const startTime = performance.now();

    const add_button = display_element.querySelector("#add");
    add_button.addEventListener("click", () => AddEx(trial.regex, (add_button as any)));

    // Add the first example box
    AddEx(trial.regex, add_button as any);

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
  input.addEventListener('change', e => updateWhetherItIsValid(regex, e.target))
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
