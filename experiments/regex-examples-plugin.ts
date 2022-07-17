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
    console.log(`Asking for examples for ${trial.regex}`);
    var html = "";
    // TODO: make it be good (like in experiment1.html)
    html += `
    <div>Please provide examples for the description below!</div>
    <p><strong>Description:</strong> <span class=description>${trial.description}<span></p>
    <p><strong>Regex:</strong> ${trial.regex}</p>
    `;
    // start form
    html += '<form id="regex-examples-form" autocomplete="off">';


    

    // add form HTML / input elements
    // TODO: add the actual input elements (like from experiment1.html)

    //html += `<p id="the_examples"> <div> <br> <button id='remove' class="remove" type="button")> Remove </button> <span> Example: </span> <input name="first" type="text" class = "example"/> <span class = "wrong"> Invalid </span>  </div> </p> <button id="add" type="button"> Add Example </button>`
    html += `<p id="the_examples"></p> <button id="add" type="button"> Add Example </button>`

    // add submit button
    html +=
      '<input type="submit" id="jspsych-survey-html-form-next" class="jspsych-btn jspsych-survey-html-form" value="Continue"></input>';

    html += "</form>";
    display_element.innerHTML = html;

    const startTime = performance.now();

    const add_button = display_element.querySelector("#add");
    add_button.addEventListener("click", () => AddEx(trial.regex, (add_button as any)));

    // Add the first example box
    AddEx(trial.regex, add_button as any);

    const this_form = display_element.querySelector("#regex-examples-form");
    this_form
      .addEventListener("submit", (event) => {
        // don't submit form
        event.preventDefault();

        if ( document.getElementsByClassName("wrong").length > 0 ) {
          alert("Not all your examples are valid >:(");
          return
        }

        // measure response time
        const endTime = performance.now();
        const response_time = Math.round(endTime - startTime);

        const question_data = objectifyForm(serializeArray(this_form));

        // save data
        var trialdata = {
          rt: response_time,
          response: question_data,
        };

        display_element.innerHTML = "";

        // next trial
        this.jsPsych.finishTrial(trialdata);
      }
      );
  }
}

/**
 * Serialize all form data into an array
 * @copyright (c) 2018 Chris Ferdinandi, MIT License, https://gomakethings.com
 * @param  {Node}   form The form to serialize
 * @return {String}      The serialized form data
 */
function serializeArray(form: any) {
  // Setup our serialized data
  var serialized = [];

  // Loop through each field in the form
  for (var i = 0; i < form.elements.length; i++) {
    var field = form.elements[i];

    // Don't serialize fields without a name, submits, buttons, file and reset inputs, and disabled fields
    if (
      !field.name ||
      field.disabled ||
      field.type === "file" ||
      field.type === "reset" ||
      field.type === "submit" ||
      field.type === "button"
    )
      continue;

    // If a multi-select, get all selections
    if (field.type === "select-multiple") {
      for (var n = 0; n < field.options.length; n++) {
        if (!field.options[n].selected) continue;
        serialized.push({
          name: field.name,
          value: field.options[n].value,
        });
      }
    }

    // Convert field data to a query string
    else if ((field.type !== "checkbox" && field.type !== "radio") || field.checked) {
      serialized.push({
        name: field.name,
        value: field.value,
      });
    }
  }

  return serialized;
}

// from https://stackoverflow.com/questions/1184624/convert-form-data-to-javascript-object-with-jquery
function objectifyForm(formArray: any) {
  //serialize data function
  var returnArray = <any>{};
  for (var i = 0; i < formArray.length; i++) {
    returnArray[formArray[i]["name"]] = formArray[i]["value"];
  }
  return returnArray;
}


/*********** Handlers for various interation events ************/

// Our functions :) 
async function RemoveEx(e : HTMLElement) {
  console.log(e);
  const div = e.parentNode;
  console.log(div);
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
  button.addEventListener('click', (e) => RemoveEx(e.target as any))
  const ex = document.createElement("span")
  //TODO: Add a way to check for empty string!
  ex.innerHTML = " Example: "
  main.appendChild(ex)
  const input = document.createElement("input")
  input.type = "text"
  //TODO: REMOVE DUPLICATE CODE
  input.addEventListener('change', (e) => updateWhetherItIsValid(regex, e.target))
  const correct = document.createElement("span")
  //TODO: Add a way to check for empty string!
  correct.innerHTML = " Invalid"
  correct.className = "wrong"
  main.appendChild(input)
  main.appendChild(correct)
  updateWhetherItIsValid(regex,input)
  const div = e.parentNode
  console.log(main)
  document.getElementById("the_examples").appendChild(main)
}

function updateWhetherItIsValid(regex: string, t: any) {
  //const target = e.target as any;
  console.log(t.value)
  console.log(t)
  const re = new RegExp("^" + regex + "$");
  console.log(regex)

  if ( re.test(t.value)) {
    console.log("I MATCH :)")
    const parent = t.parentNode
    const feedback = parent.getElementsByTagName("span")[1]
    feedback.innerHTML = "Valid"
    feedback.className  = "correct"
    //e.target.classList.add("correct") 
  } else {
    console.log("Not matching :(");
    const parent = t.parentNode
    const feedback = parent.getElementsByTagName("span")[1]
    feedback.innerHTML = " Invalid"
    feedback.className  = "wrong"
    //if (e.target.classList[1] == "correct"){
    //e.target.classList.remove('correct')
    //}
  }
}