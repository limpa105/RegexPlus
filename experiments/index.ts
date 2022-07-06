// This is the main typescript file that runs in the browser.
//
// (It imports stuff from the other typescript files.)

import {initJsPsych} from 'jspsych';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import {RegexExamplesPlugin, RegexExampleData} from './regex-examples-plugin';

const jsPsych = initJsPsych({
  /* options go here */
});

/*********** The main content ***********/
/* We could change it to get the content from a server, or from somewhere else,
 * or whatever, in the future
 */
// TODO
const problems: RegexExampleData[] = [
  {
    description: "lotsa A's",
    regex: "A*a+",
  },
];

/*********** Intro slides ***********/
const welcome = {
  type: htmlKeyboardResponse,
  stimulus: `
    <p>Welcome to the regular expressions experiment!</p>
    <p>Press any key to continue.</p>
  `
};
const instructions = {
  type: htmlKeyboardResponse,
  stimulus: `
    <p>In this experiment, you will be shown descritpions of different groups of strings. </p>
    <p> <strong> For each description your goal is to provide examples of strings that fit the description and allow someone who has never seen the description to guess it based solely on the examples. </strong> </p>
    <p> You will be given 20 blank text boxes. Please put each example into a separate box. If your example fits the string it will turn green; otherwise it will remain red. You may provide as many or as few examples as you deem necessary to describe the description.</p>
    <p>Press any key to continue.</p>
  `
};
const example = {
  type: htmlKeyboardResponse,
  stimulus: `
    <p> Below is an example of the task </p>
    <p> <br> <br> <stong> Description </strong>:All strings made up of only capital letters and the empty string
  <br> <stong>  Corresponding Regex </strong>: [A-Z]* </p>
  <ol>
    <li> </li>
    <li> A </li>
    <li> BOLD </li>
    <li> REGEXES </li>
  </ol>

  <p>Press any key to begin.</p>
  `
};

const thank = {
  type: htmlKeyboardResponse,
  stimulus: `<p> Thank you for partcipating in our study! </p>`
};

/************* Putting it all together *************/

const main_experiment = {
  timeline: [{
    type: RegexExamplesPlugin,
    description: jsPsych.timelineVariable('description'),
    regex: jsPsych.timelineVariable('regex'),
  }],
  timeline_variables: problems,
};

const timeline = [
  welcome,
  instructions,
  example,
  main_experiment,
  thank,
];

jsPsych.run(timeline);

