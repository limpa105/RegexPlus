import { JsPsych, JsPsychPlugin, ParameterType, TrialType } from "jspsych";
import SurveyTextPlugin from './survey-text';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';


const info = <const>{
  name: "load-more-stuff",
  parameters: {
  userID: {
      type: ParameterType.STRING,
      pretty_name: "user_id",
      default: "something went horribly wrong",
  },
  data_id: {
      type: ParameterType.INT,
      pretty_name: "personna id",
      default: 0,
    }, 
  },
};

type Info = typeof info;

function listener_boilerplate(ex: string[]) : string {
  return `
  <p> As a reminder, our regular expression grammar allows: 
  <ol> 
  <li> Character classes: <code>[a-z]</code>, <code>[0-9]</code>, <code>[A-Z]</code>, <code>[a-zA-Z]</code>, <code>[a-zA-Z0-9]</code> </li>
  <li> Optionals: <code>(wow)?</code> or <code>([a-z])?</code> </li>
  <li> Repetitions of the character classes: <code>[a-z]+</code> or <code>[a-z]*</code> or <code>[a-z]{3}</code> </li>
  </p>
  <p>
  Try to guess the regular expression that describes the examples below:
  <br> <ol class = 'examples_list'>${ex.map(x => '<li><code>' + x + '</code></li>').join(' ')}</ol> <br>
  </p>
  `;   
}

const thank = {
  type: htmlKeyboardResponse,
  stimulus: `<p> Thank you for partcipating in our study! </p>`
};

class LoadMoreStuffPlugin implements JsPsychPlugin<Info> {
  static info = info;

  constructor(private jsPsych: JsPsych) {}

  
  trial(display_element: HTMLElement, trial: TrialType<Info>) {
    // load data
    // TODO: load data from the server
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '/data_for/' + trial.data_id, false);
    xhr.send(null);
    if (xhr.status !== 200) {
      alert('Could not connect to server');
      return;
    }
    const new_data = JSON.parse(xhr.responseText);
    if (new_data.all_done) {
      console.log("We finished! :)")
      this.jsPsych.finishTrial();
      this.jsPsych.addNodeToEndOfTimeline({ timeline: [thank] });
      this.jsPsych.finishTrial();
      return;
    }
    this.jsPsych.addNodeToEndOfTimeline({
      timeline: [
        // First a trial with the actual thing
        {
          type: SurveyTextPlugin,
          questions : [
            {prompt: listener_boilerplate(new_data.examples)}
          ],
          regex: new_data.regex,
          userID: trial.userID,
          data_id: trial.data_id,
          example_id: new_data.example_id
        },
        // Then a "trial" to load more data
        {
          type: LoadMoreStuffPlugin,
          userID: trial.userID,
          data_id: trial.data_id,
        }
        // Oh my gosh this is so hacky
        // vibes
      ],
    });
    this.jsPsych.finishTrial({});
  }
}

export default LoadMoreStuffPlugin;

/*

Data Needs to have 
regex:
example_id:
examples


[{1:5}, {2:3}, {3:10}]
[{2:3},{1:5} [1]....]
# reset start at begining if value!=10
# put up a screen # Experiment is over





1 ---> 5
2 ---> 




*/