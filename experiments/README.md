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
build tools, and dumps `index.html` and `bundle.js` into `build/`. Open
`build/index.html` in your browser.

To run a server: haven't figured out this part yet.


