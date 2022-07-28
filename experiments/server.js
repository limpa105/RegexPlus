const fs = require('fs');
const express = require('express');
const app = express();

const port_no = 8888;

const web_dir = __dirname + '/build';
const saved_data_dir = __dirname + '/data';

const server = app.listen(port_no);
console.log(`Listening for HTTP on ${port_no}...`);

app.use(express.json());
app.use(express.urlencoded());

app.get('/*', (req, res) => {
  const fileName = req.params[0];
  console.log('\tGET: file requested:', fileName);
  if (fileName === '')
    return res.redirect('/index.html');
  return res.sendFile(fileName, { root: web_dir });
});

app.post('/have_some_data', (req, res) => {
  // TODO: save the data (to a database, or even just a file)
  console.log("\tPOST: Got some data:", req.body);
  const { prolificID, studyID, sessionID, data } = req.body;
  const fileName = `from_user_${prolificID}.json`;
  try {
    fs.writeFileSync(saved_data_dir + '/' + fileName, JSON.stringify(req.body));
      // file written successfully
    console.log("Have some files")
  } catch (err) {
    console.error('Error writing file:', err);
  }
  res.sendStatus(200);
});


