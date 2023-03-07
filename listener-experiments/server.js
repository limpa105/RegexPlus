const fs = require('fs');
const express = require('express');
const app = express();

const port_no = 8755;

const web_dir = __dirname + '/build';
const saved_data_dir = __dirname + '/data';

const server = app.listen(port_no);
console.log(`Listening for HTTP on ${port_no}...`);

app.use(express.json());
app.use(express.urlencoded());

/** Server-side data:
 *
 * for each slice of data, for each set of examples, a flag indicating not started, in progress, finished
 */
const NOT_STARTED = 'Not started';
const IN_PROGRESS = 'In progress';
const FINISHED = 'Finished';

const state = {};
const NUM_USERS = 11;
const NUM_REGEXES = 21;
for (let id = 0; id < NUM_USERS; id++) {
  state[id] = [];
  for (let ex_id = 0; ex_id < NUM_REGEXES; ex_id++)
    state[id][ex_id] = NOT_STARTED;
}

function next_id() {
  // Rule for choosing which ID to assign: whichever id has the most NOT_STARTED problems.
  // Ties broken by whichever has the fewest FINISHED problems.
  const not_started_counts = [];
  const in_progress_counts = [];
  for (let i = 0; i < NUM_USERS; i++) {
    not_started_counts[i] = 0;
    for (let j = 0; j < NUM_REGEXES; j++) {
      not_started_counts[i] += (state[i][j] === NOT_STARTED);
      in_progress_counts[i] += (state[i][j] === IN_PROGRESS);
    }
  }
  
  let cur_best = 0;
  for (let i = 0; i < NUM_USERS; i++) {
    if (not_started_counts[i] > not_started_counts[cur_best] ||
      (not_started_counts[i] == not_started_counts[cur_best] && in_progress_counts[i] > in_progress_counts[cur_best]))
      cur_best = i;
  }
  return cur_best;
}

function next_index_for_id(id) {
  // Rule for choosing which example index to assign: whichever example is NOT_STARTED if any.
  // If they are all started then whichever is IN_PROGRESS
  // If there are none left, we'll send something that states no more data
  let i = 0;
  for (; i < NUM_REGEXES; i++)
    if (state[id][i] == NOT_STARTED) return i;
  for (i = 0; i < NUM_REGEXES; i++)
    if (state[id][i] == IN_PROGRESS) return i;
  return undefined;
}

app.get('/whoami', (req, res) => {
  console.log("Assigning user id")
  const id = next_id();
  return res.send({ id: id });
});

app.get('/data_for/:id', (req, res) => {
  const id = parseInt(req.params.id);
  console.log("Getting data for id", id)
  const contents = JSON.parse(fs.readFileSync('data_' + id + '.json'));
  const example_id = next_index_for_id(id)
  if (example_id === undefined) {
    console.log("There are no more examples left for this id")
    return res.send({ all_done: true });
  }
  state[id][example_id] = IN_PROGRESS;
  res.send({
    all_done: false,
    regex: contents.regex[example_id],
    example_id,
    examples: JSON.parse(contents.examples[example_id]),
  })
});

app.get('/*', (req, res) => {
  const fileName = req.params[0];
  console.log('\tGET: file requested:', fileName);
  if (fileName === '')
    return res.redirect('/index.html');
  return res.sendFile(fileName, { root: web_dir });
});

app.post('/have_some_data', (req, res) => {
  console.log("\tPOST: Got some data:", req.body);
  const { user_id, regex, data_id, example_index, data} = req.body;
  const fileName = `from_user_${user_id}_at_${new Date()}.json`;
  const dirName = `${saved_data_dir}/from_user_${user_id}`
  // Update the state matrix
  state[data_id][example_index] = FINISHED;
  // save the data to a JSON file
  try {
    fs.mkdirSync(dirName, {recursive: true}); // "recursive" actually means: no error if it already exists
    fs.writeFileSync(dirName + '/' + fileName, JSON.stringify(req.body));
    // file written successfully
    console.log("Have some files")
  } catch (err) {
    console.error('Error writing file:', err);
  }
  res.sendStatus(200);
});


