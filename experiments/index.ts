// This is the main typescript file that runs in the browser.
//
// (It imports stuff from the other typescript files.)

import {initJsPsych} from 'jspsych';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import {RegexExamplesPlugin, RegexExampleData} from './regex-examples-plugin';

var queryString = window.location.search;
var urlParams = new URLSearchParams(queryString);
var prolificID = urlParams.get("PROLIFIC_PID"); // ID unique to the participant
var studyID = urlParams.get("STUDY_ID"); // ID unique to the study
var sessionID = urlParams.get("SESSION_ID"); // ID unique to the particular submission
var projName = urlParams.get("projName");
var expName = urlParams.get("expName");
var iterName = urlParams.get("iterName");


var jsPsych = initJsPsych({
  /* options go here */
  on_data_update: function() {
    saveData(jsPsych.data.get().csv());
  },
});

function saveData(data: string) {
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/have_some_data');
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send(JSON.stringify({ prolificID, studyID, sessionID, data }));
}

/*********** The main content ***********/
/* We could change it to get the content from a server, or from somewhere else,
 * or whatever, in the future
 */
// TODO
const problems: RegexExampleData[] = [
  {
    description: "All nonempty lowercase strings",
    regex: "[a-z]+",
  },
  {
    description: "All strings made up of uppercase letters, lowercase letters and digits, and that contain at least one lowercase letter",
    regex: "[A-Za-z0-9]*[a-z][A-Za-z0-9]*",
  },
  {
    description: "All strings made up of “<tt>  09 </tt> ” followed by 7 digits",
    regex: "09\\d{7}",
  },
  {
    description: "All strings made up of “<tt>Page  </tt>” followed by at least one digit followed by “<tt>  of </tt>” and at least one digit",
    regex: "Page \\d+ of \\d+",
  },
  {
    description: "All strings made up of “<tt>abc.</tt>” followed by at least one digit",
    regex: "abc\\.\\d+",
  },
  {
    description: "All strings made up of 6 digits",
    regex: "\\d{6}",
  },
  {
    description: "All strings made up of 6 digits followed by “<tt>.</tt> ” followed by 3 digits",
    regex: "\\d{6}\\.\\d{3}",
  },
  {
    description: "All strings made up of 3 digits followed by “<tt>-</tt> ” followed by 3 digits followed by “<tt>-</tt>” followed by 4 digits",
    regex: "\\d{3}-\\d{3}-\\d{4}",
  },
  {
    description: "All strings made up of at least one digit optionally followed by “<tt>.</tt>” followed by any number of digits",
    regex: "\\d+(\\.)?\\d*",
  },
  {
    description: "All strings made up of 1 to 3 lowercase letters followed by “<tt>-</tt>” followed by 1 to 2 lowercase letters followed by “<tt>-</tt>” followed by 1 to 4 digits",
    regex: "[a-z]{1,3}-[a-z]{1,2}-\\d{1,4}",
  },
  {
    description: "All strings made up of at least one digit followed by “<tt>.</tt>” followed by 1 to 4 digits",
    regex: "\\d+\\.\\d{1,4}",
  },
  {
    description: "All strings made up of an optional “<tt>*</tt>” followed by at least two lowercase letters followed by an optional “<tt>*</tt>”",
    regex: "(\\*)?[a-z]{2,}(\\*)?",
  },
  {
    description: "All strings made up of one or more digits, with an optional “<tt>+</tt>” in front",
    regex: "(\\+)?\\d+",
  },
  {
    description: "All strings made up of two digits followed by “<tt>.5</tt>”",
    regex: "\\d{2}\\.5",
  },
  {
    description: "All strings made up of “<tt>C0</tt>” followed by four digits",
    regex: "C0\\d{4}",
  },
  {
    description: "All strings made up of either one lowercase OR one uppercase letter followed by five digits",
    regex: "[a-zA-Z]\\d{5}",
  },
  {
    description: "All strings made up of an uppercase letter followed by one or more lowercase letters, then a space, then another uppercase letter followed by one or more lowercase letters",
    regex: "[A-Z][a-z]+ [A-Z][a-z]+",
  },
  {
    description: "All nonempty strings made up of “9” followed by 9 digits",
    regex: "9\\d{9}"
  },
  {
    description: "All nonempty strings made up of digits, lowercase and uppercase letters",
    regex: "[a-zA-Z0-9]+"
  },
  {
    description: "All nonempty strings made up of digits",
    regex: "\\d+"
  },
  {
    description: "All strings made up of 3 digits followed by a space followed by 2 digits",
    regex: "\\d{3} \\d{2}"
  },
];

/*********** Intro slides ***********/
const welcome = {
  type: htmlKeyboardResponse,
  stimulus: `
    <p>Welcome to this experiment!</p>
    <p>By continuing, you are consenting to participate in it.</p>
    <p>Press any key to continue.</p>
  `
};
const instructions = {
  type: htmlKeyboardResponse,
  stimulus: `
   <p>In this experiment, you will be shown descriptions of different groups of strings. Each description corresponds to a regular expression (regex).</p>
    <p> <strong> For each description your goal is to provide examples of strings that would allow someone who has never seen the description to guess it based solely on the examples. </strong> </p>
    <p> <br> For example: </p>
    <div class=task_ex>
    <p>  <strong>Description:</strong> All strings made up of only capital letters 
    <br> <strong>Corresponding Regex:</strong> <tt> [A-Z]*  </tt> </p>
    <ol>
      <li> </li>
      <li> I </li>
      <li> LOVE </li>
      <li> REGEXES </li>
    </ol>
    </div class=task_ex>
    <p>Press any key to continue.</p>
    </div class=task_ex>
  `
};
const example = {
  type: htmlKeyboardResponse,
  stimulus: `
  <p> Press a button to either add or remove examples. If your example fits the description it will have "Valid" appear next to it in green, and if it doesn't it will be labelled "Invalid".</p>
  <p> All of your examples need to be valid. </p> 
  <p> You may provide as many or as few examples as you deem necessary to convey the description.</p>
  <p>Press any key to begin.</p>
  `
};


const thank = {
  type: htmlKeyboardResponse,
  stimulus: `<p> Thank you for partcipating in our study! </p>
  <p> To get credit please give Prolific this code: <strong>ABCDEFGH</strong></p>`
};

/************* Putting it all together *************/

const main_experiment = {
  timeline: [{
    type: RegexExamplesPlugin,
    description: jsPsych.timelineVariable('description'),
    regex: jsPsych.timelineVariable('regex'),
  }],
  timeline_variables: problems,
  randomize_order: true,
  
};

const timeline = [
  welcome,
  instructions,
  example,
  main_experiment,
  thank,
];

jsPsych.run(timeline);

