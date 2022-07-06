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
    <div>this is the intro to this problem</div>
    <p><strong>Description:</strong> <span class=description>${trial.description}<span></p>
    <p><strong>Regex:</strong> <pre>${trial.regex}</pre></p>
    `;
    // start form
    html += '<form id="regex-examples-form" autocomplete="off">';

    // add form HTML / input elements
    // TODO: add the actual input elements (like from experiment1.html)

    // add submit button
    html +=
      '<input type="submit" id="jspsych-survey-html-form-next" class="jspsych-btn jspsych-survey-html-form" value="Continue"></input>';

    html += "</form>";
    display_element.innerHTML = html;

    const startTime = performance.now();

    const this_form = display_element.querySelector("#regex-examples-form");
    this_form
      .addEventListener("submit", (event) => {
        // don't submit form
        event.preventDefault();

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
      });
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
