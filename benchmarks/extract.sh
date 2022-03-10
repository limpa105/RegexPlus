#!/bin/bash

# For extracting the benchmarks from https://github.com/utopia-group/regel

cat >examples.py <<EOF
class Example:
    def __init__(self, *args):
        self.inputs = args[:-1]
        self.regex = args[-1]

all_benchmarks = [
EOF

for file in benchmark/* ; do
  ./extract.sed < $file >> examples.py
done

sed -i '$s/,$/]/' examples.py


