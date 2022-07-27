const express = require('express');
const app = express();

const port_no = 8888;

const server = app.listen(port_no);
console.log(`Listening for HTTP on ${port_no}...`);

app.use(express.json());
app.use(express.urlencoded());

app.get('/*', (req, res) => {
  const fileName = req.params[0];
  console.log('\tGET: file requested:', fileName);
  if (fileName === '')
    return res.redirect('/index.html');
  return res.sendFile(fileName, { root: __dirname + '/build' });
});

app.post('/have_some_data', (req, res) => {
  // TODO: save the data (to a database, or even just a file)
  console.log("\tPOST: Got some data:", req.body);
  res.sendStatus(200);
});


