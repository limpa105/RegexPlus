# Experiment setup

- `index.html` has the main HTML that just loads static resources + JS files
- `index.ts` does all the work callincdg `jsPsych` and setting it up
- `regex-examples-plugin.ts` has the code for a single "gimme examples" question
- `experiment1.html` is just there for reference (it is not used)

Once, after cloning the repo: run `npm install`.

Also needed to:
 - `npm install webpack`
 - 

After a change, to rebuild: run `d`. This runs all the
build tools, and dumps `index.html` and `bundle.js` into `build/`.

To run a server, locally:
 - Remember to rebuild with `npm run build-web`!
 - Run `node server.js`
 - Go to `http://localhost:8888/index.html?PROLIFIC_PID=test_id` in your browser

Pre-launch checklist:
 - [ ] There is an empty folder `data/` in the current directory
 - [ ] `index.ts` has the Prolific completion code
 - [ ] Run `npm run build-web`!
 - [ ] You are in a tmux environment so that it keeps running after you leave ssh
 - [ ] Run `node server.js`
 - [ ] You detach from tmux/screen using CTRL + b then d 
 - [ ] Experiment works with link once logged out of server
 - [ ] Experiment saves data on the server using the link on the server
 - [ ] To check that the folder is correct use ?PROLIFIC_PID=test_id

