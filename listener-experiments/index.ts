// This is the main typescript file that runs in the browser.
//
// (It imports stuff from the other typescript files.)

import {initJsPsych} from 'jspsych';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import instructions from '@jspsych/plugin-instructions';
import surveyMulipleChoice from '@jspsych/plugin-survey-multi-choice';
import SurveyTextPlugin from './survey-text';
import LoadMoreStuffPlugin from './load-more-stuff';

var queryString = window.location.search;
var urlParams = new URLSearchParams(queryString);
var prolificID = urlParams.get("PROLIFIC_PID") ?? "unknown_user_" + Math.floor(1000000000 * Math.random()); // ID unique to the participant
var studyID = urlParams.get("STUDY_ID"); // ID unique to the study
var sessionID = urlParams.get("SESSION_ID"); // ID unique to the particular submission

var jsPsych = initJsPsych({
  /* options go here */
  // on_data_update: function() {
  //   saveData(jsPsych.data.get().csv());
  // },
  show_progress_bar: false,
  message_progress_bar: 'Progress Bar',
  auto_update_progress_bar: false,
});

/*********** Intro slides ***********/

const consentHTML = ["<u><p id='legal'>Consent to Participate</p></u>\
    <p id='legal'>By completing this HIT, you are participating in a \
  study being performed by cognitive scientists in the UC San Diego \
  Department of Psychology. The purpose of this research is to find out\
  how people communicate about patterns. \
  You must be at least 18 years old to participate. There are neither\
  specific benefits nor anticipated risks associated with participation\
  in this study. Your participation in this study is completely voluntary\
  and you can withdraw at any time by simply exiting the study. You may \
  decline to answer any or all of the following questions. Choosing not \
  to participate or withdrawing will result in no penalty. Your anonymity \
  is assured; the researchers who have requested your participation will \
  not receive any personal information about you, and any information you \
  provide will not be shared in association with any personally identifying \
  information.</p> \
  <p> If you have questions about this research, please contact the \
  researchers by sending an email to \
  <b><a href='mailto://cogtoolslab.requester@gmail.com'>cogtoolslab.requester@gmail.com</a></b>. \
  These researchers will do their best to communicate with you in a timely, \
  professional, and courteous manner. If you have questions regarding your \
  rights as a research subject, or if problems arise which you do not feel \
  you can discuss with the researchers, please contact the UC San Diego \
  Institutional Review Board.</p><p>Click 'Next' to continue \
  participating in this HIT.</p>"];



const describeHTML2 = [ `<p>
    In this study you will try to guess regular expressions from examples </p>
    <p> Our regular expression grammar allows: 
    <ol> 
    <li> The following character classes : [a-z], [0-9], [A-Z], [a-zA-Z], [a-zA-Z0-9] </li>
    <li> Optional of a string or ONE character class from above. (Ex: <code>(wow)?</code> or <code>([a-z])?</code >) </li>
    <li> Repetitions of the character classes above. (Ex: <code>[a-z]+</code> or <code>[a-z]*</code> or <code>[a-z]{3}</code>) </li>
    </p>
    <p>
    For example, the following regexes are in our grammar:
    <ol>
      <li><code>[a-z]+</code></li>
      <li><code>[A-Z]*[0-9]*</code></li>
      <li><code>hello( friends)?</code></li>
      <li><code>[0-9]{3}-[0-9]{3}-[0-9]{4}</code></li>
    </ol>
    </p>
    <p> The following are NOT in our grammar: </p>
    <p>
    <ol >
      <li><code>[0-9]+(\\.[0-9]+)?</code> — it has the complex optional <code>(\\.[0-9]+)?</code></li>
      <li><code>[A-Z]{3}([0-9]{2})?</code> — it has the complex optional <code>([0-9]{2})?</code></li>
      <li><code>(cat|dog)</code> — it has an OR of two strings</li>
      <li><code>(cat)+</code> — it has a repetition of a string</li>
    </ol>
    </p>
    <p>
    It is not crucial that you memorize the grammar by heart as long as you remember the main principles.</p>
    `]

  

const listenerInstructionsHTML = [' <p> Type in the regular expression into the designated box and press the "Guess" button! </p>']



/** 
const train2 = {
  type: surveyMulipleChoice,
  questions: [
    {
      prompt: "Choose the regular expression that the examples below are describing: <br> <ol> <li>111.</li> <li>111.1</li> </ol>",
      options: ["111(.1)?", "111.(1)?", "[0-9]{3}.[0-9]*", "[0-9]+.[0-9]+", "[0-9]*.[0-9]*", "[0-9].(1)?", "(111.)?(1)?"],
    },]
}
*/

const listener_boilerplate = (ex) => `
    <p> As a reminder, our regular expression grammar allows: 
    <ol> 
    <li> Character classes: <code>[a-z]</code>, <code>[0-9]</code>, <code>[A-Z]</code>, <code>[a-zA-Z]</code>, <code>[a-zA-Z0-9]</code> </li>
    <li> Optionals: <code>(wow)?</code> or <code>([a-z])?</code> </li>
    <li> Repetitions of the character classes: <code>[a-z]+</code> or <code>[a-z]*</code> or <code>[a-z]{3}</code> </li>
    </p>
    <p>
    Try to guess the regular expression that describes the examples below:
    <br> <ol class = 'examples_list'>${ex}</ol> <br>
    </p>
    `;

const ex1 = {
  type: SurveyTextPlugin,
  questions : [
  {prompt: listener_boilerplate("<li><code>1234hello1234</code></li> <li><code>78hello21</code></li>")}
  ],
  regex: '[0-9]+hello[0-9]+'
}

const ex2 = {
  type: SurveyTextPlugin,
  questions : [
  {prompt: listener_boilerplate("<li><code>APM 2402</code></li> <li><code>APM 7218</code></li> <li><code>MOS 0113</code></li> <li><code>YORK 3000</code></li>")}
  ],
  regex: '[A-Z]+ [0-9]{4}'
}

const ex3 = {
  type: SurveyTextPlugin,
  questions : [
  {prompt: listener_boilerplate("<li><code>cat</code></li> <li><code>dog</code></li> <li><code>tom</code></li> <li><code>the</code></li> <li><code>bug</code></li>")}
  ],
  regex:'[a-z]+'
}

const ex4 = {
  type: SurveyTextPlugin,
  questions : [
  {prompt: listener_boilerplate("<li><code>abc</code></li> <li><code>abc4</code></li>")}
  ],
  regex: '[a-z]{3}([0-9])?'
}

const welcome = {
  type: instructions,
  pages: [ 
    consentHTML,
    describeHTML2,
    listenerInstructionsHTML,
  ],
    show_clickable_nav: true
};




/** 
const train3 = {
  type: FreeSortPlugin,
  stimuli: regexes,
  stim_width: 90,
  stim_height: 70,
  sort_area_width: 700,
  sort_area_height: 300,
  prompt: "Try to guess the regular expression the examples below are describing: <br> <ol> <li>111</li> <li>111.1</li> </ol> <br> Drag the corresponding regex blocks into the red area and put them into the correct order. <br> <br> "
}
*/
const prior_knowledge = {
  type: surveyMulipleChoice,
  preamble: 'Thank you! We have two last questions for you',
  questions: [
    {
      prompt: "I write computer programs:",
      name: "programming_experience",
      options: ["Occasionally", "Professionally"],
      required: false,
    },
    {
      prompt: "I read or write regular expressions:",
      name: "regex_experience",
      options: ["Never", "Occasionally", "Regularly"],
      required: false,
    },
  ],
};

/************* Putting it all together *************/



/*const timeline = [
  welcome,
  ex1,
  ex2,
  ex3,
  ex4,
  prior_knowledge,
  thank,
];*/

const xhr = new XMLHttpRequest();
xhr.open('GET', '/whoami', false);
xhr.send(null);
if (xhr.status !== 200) {
  alert('Could not connect to server');
  throw 'panic';
}
const data_id = JSON.parse(xhr.responseText).id;

jsPsych.run([
  welcome,
  {
    type: LoadMoreStuffPlugin,
    userID: Math.floor(1000000000 * Math.random()),
    data_id: data_id,
  },
]);
