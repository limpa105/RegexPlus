# Experiment setup

- `index.html` has the main HTML that just loads static resources + JS files
- `index.ts` does all the work callincdg `jsPsych` and setting it up
- `regex-examples-plugin.ts` has the code for a single "gimme examples" question
- `experiment1.html` is just there for reference (it is not used)

Once, after cloning the repo: run `npm install`.

Also needed to:
 - `npm install webpack`
 - 

After a change, to rebuild: run `npm run build-web`. This runs all the
build tools, and dumps `index.html` and `bundle.js` into `build/`.

To run a server, locally:
 - Remember to rebuild with `npm run build-web`!
 - Run `node server.js`
 - Go to `http://localhost:8888/index.html` in your browser


