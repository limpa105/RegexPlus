// This is the main typescript file that runs in the browser.
//
// (It imports stuff from the other typescript files.)

import {initJsPsych} from 'jspsych';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import instructions from '@jspsych/plugin-instructions';
import surveyMulipleChoice from '@jspsych/plugin-survey-multi-choice';
import SurveyTextPlugin from './survey-text';
import {RegexExamplesPlugin, RegexExampleData} from './regex-examples-plugin';
import FreeSortPlugin from './drag-and-drop';

var queryString = window.location.search;
var urlParams = new URLSearchParams(queryString);
var prolificID = urlParams.get("PROLIFIC_PID") ?? "unknown_user_" + Math.floor(1000000000 * Math.random()); // ID unique to the participant
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
  show_progress_bar: true,
  message_progress_bar: 'Progress Bar',
  auto_update_progress_bar: false,
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
const problems: RegexExampleData[] = [
  {
    description: "All nonempty lowercase strings",
    regex: "[a-z]+",
  },
  {
    description: "Alphanumeric strings that contain at least one lowercase letter",
    regex: "[A-Za-z0-9]*[a-z][A-Za-z0-9]*",
  },
  {
    description: "“<tt>09</tt>” followed by 7 digits",
    regex: "09[0-9]{7}",
  },
  {
    description: "“<tt>Page </tt>” followed by at least one digit followed by “<tt> of </tt>” and at least one digit",
    regex: "Page [0-9]+ of [0-9]+",
  },
  {
    description: "“<tt>abc.</tt>” followed by at least one digit",
    regex: "abc\\.[0-9]+",
  },
  {
    description: "6 digits",
    regex: "[0-9]{6}",
  },
  {
    description: "6 digits followed by “<tt>.</tt>” followed by 3 digits",
    regex: "[0-9]{6}\\.[0-9]{3}",
  },
  {
    description: "3 digits followed by “<tt>-</tt>” followed by 3 digits followed by “<tt>-</tt>” followed by 4 digits",
    regex: "[0-9]{3}-[0-9]{3}-[0-9]{4}",
  },
  {
    description: "At least one digit optionally followed by “<tt>.</tt>” followed by any number of digits",
    regex: "[0-9]+(\\.)?[0-9]*",
  },
  {
    description: "1 to 3 lowercase letters followed by “<tt>-</tt>” followed by 1 to 2 lowercase letters followed by “<tt>-</tt>” followed by 1 to 4 digits",
    regex: "[a-z]{1,3}-[a-z]{1,2}-[0-9]{1,4}",
  },
  {
    description: "At least one digit followed by “<tt>.</tt>” followed by 1 to 4 digits",
    regex: "[0-9]+\\.[0-9]{1,4}",
  },
  {
    description: "An optional “<tt>*</tt>” followed by at least two lowercase letters followed by an optional “<tt>*</tt>”",
    regex: "(\\*)?[a-z]{2,}(\\*)?",
  },
  {
    description: "One or more digits, with an optional “<tt>+</tt>” in front",
    regex: "(\\+)?[0-9]+",
  },
  {
    description: "Two digits followed by “<tt>.5</tt>”",
    regex: "[0-9]{2}\\.5",
  },
  {
    description: "“<tt>C0</tt>” followed by four digits",
    regex: "C0[0-9]{4}",
  },
  {
    description: "A letter followed by five digits",
    regex: "[a-zA-Z][0-9]{5}",
  },
  {
    description: "An uppercase letter followed by one or more lowercase letters, then a space, then another uppercase letter followed by one or more lowercase letters",
    regex: "[A-Z][a-z]+ [A-Z][a-z]+",
  },
  {
    description: "“9” followed by 9 digits",
    regex: "9[0-9]{9}"
  },
  {
    description: "All nonempty alphanumeric strings",
    regex: "[a-zA-Z0-9]+"
  },
  {
    description: "All nonempty strings made up of digits",
    regex: "[0-9]+"
  },
  {
    description: "3 digits followed by a space followed by 2 digits",
    regex: "[0-9]{3} [0-9]{2}"
  },
];
const NUM_TASKS = problems.length;


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

const storyHTML = [`
    <p>In this study you will be a space explorer traveling from planet to
    planet in a new solar system called Gacradus. This solar system has a very
    elaborate etiquette system. On every planet, there are strict rules for
    which words you are allowed to use to talk to the inhabitants. You are
    corresponding with your friend Charlie on Earth, who is charged with
    documenting the intricate language system of this new solar system. Your
    goal is to communicate all of the details about the language of each planet
    to Charlie, but Gacradians are screening your letters and only allowing you
    to send words that are in their language. </p>
    <p>Become a world-famous explorer by explaining all of the details of each
    planet’s language through examples of allowed words! </p>
    <img src="images/conversation.png" width="450" height="300"<br>
    `];

const describeHTML2 = [ `<p>
    In this study you will try to guess regular expressions from examples and provide examples that will allow others to guess regular expressions. </p>
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

  

const listenerInstructionsHTML = ['<p> First you will try to guess regular expressions based on examples someone else gave. </p>  <p> Type in the regular expression into the designated box and press the "Guess" button! </p>']

const speakerInstructionsHTML = [`<p> Now you will try to provide examples so someone else can guess the regular expression from them. </p>
<p> Please do not communicate the regular expression using actual words. For example for [A-Z]+ do not say ALLCAPITALLETTERS.</p>  
<p> Press a button to either add or remove examples. \
If your example fits the description it will have "Valid" appear next to it in green, \
and if it doesn\'t it will be labelled "Invalid".</p> \
<p> All of your examples need to be valid. </p>  \
<p> You may provide as many or as few examples as you deem necessary to convey the description.</p> \
<p>Click next to begin the experiment!</p> `]

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

const more_instructions = {
  type: instructions,
  pages: [ 
    speakerInstructionsHTML
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
    NUM_TASKS: NUM_TASKS,
  }],
  timeline_variables: problems,
  randomize_order: true,
  
};

const timeline = [
  welcome,
  ex1,
  ex2,
  ex3,
  ex4,
  more_instructions,
  main_experiment,
  prior_knowledge,
  thank,
];


jsPsych.run(timeline);

