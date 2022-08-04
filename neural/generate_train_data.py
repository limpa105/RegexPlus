import next_chars
NUM_EXAMPLES = 10

with open('train_data.txt') as f:
    lines = f.readlines()

regexes = [eval(line.strip()) for line in lines]
print(len(regexes))


with open('train_data_pars.txt', 'w') as f:
	count = 0
	for regex in regexes: 
		nfa = next_chars.regex_to_nfa(regex)
		dfa = next_chars.DFA(nfa)
		for i in range(NUM_EXAMPLES):
			example = dfa.sample()
			f.write("%s\n" % example)
		count+=1
		if (count%1000 == 0):
			print(count)
	print(count)



