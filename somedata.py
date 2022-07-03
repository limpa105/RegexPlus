# import rstr
import hypothesis
import warnings
warnings.filterwarnings("ignore")

benchmarks = []

all_inputs = []
def inputs(s):
    global all_inputs
    all_inputs += s.split(', ')
    benchmarks.append(s.split(', '))

all_regexes = [
    r'[A-Za-z0-9]*[a-z][A-Za-z0-9]*',
    r'[a-z]{1,3}-[a-z]{1,2}-\d{1,4}',
    r'\d+.\d{1,4}',
    r'(-)?\d+(,)?\d{0,2}',
    r'9\d{9}',
]
NUM_EXTRA_EXAMPLES_PER_REGEX = 10
def regexes(s):
    global all_regexes
    for r in s.strip().split('\n'):
        all_regexes.append(r)

        g = hypothesis.strategies.from_regex('^' + r + '$')
        for i in range(NUM_EXTRA_EXAMPLES_PER_REGEX):
            all_inputs.append(g.example().strip())


inputs("t, alex, ramon, bob")
regexes(r"""
[a-z]+
[a-z]*
[a-z]*[A-Za-z]
[A-Za-z][a-z]*
[a-z]*[0-9A-Za-z]
""")
inputs("a, ChelloB, 25HbyeW98, Q78red66K")
regexes(r"""
[0-9A-Za-z]+
[0-9A-Za-z]*
(25)?[0-9A-Za-z]+
(Ch)?[0-9A-Za-z]+
(Q7)?[0-9A-Za-z]+
""")
inputs("091239567, 098764321, 093334445, 094388270")
regexes(r"""
09\d+
09\d{7}
09\d{2,}
0\d+
09\d{3,}
""")
inputs("Page 2 of 20, Page 18 of 44, Page 107 of 109, Page 7 of 0")
regexes(r"""
Page \d+ of \d+
Page \d+ of \d{1,3}
""")
inputs("abc.0, abc.123, abc.89, abc.3")
regexes(r"""
abc\.\d+
abc\.\d*
abc\.\d{1,3}
[a-z]bc\.\d+
a[a-z]c\.\d+
""")
inputs("879234, 236784, 028977, 111111")
regexes(r"""
\d{6}
\d+
\d{2,}
\d{3,}
\d{4,}
""")
inputs("010022.500, 125893.234, 314159.256, 358979.323")
regexes(r"""
\d{6}\.\d{3}
\d+\.\d{3}
\d{2,}\.\d{3}
\d{3,}\.\d{3}
\d{4,}\.\d{3}
""")
inputs("202-918-2132, 678-977-0459, 205-521-0797, 271-828-1828")
regexes(r"""
\d{3}-\d{3}-\d{4}
\d{3}-\d{3}-\d+
\d+-\d{3}-\d{4}
\d{3}-\d+-\d{4}
\d{3}-\d{3}-\d{2,}
""")
inputs("0, 17.2, 8., 100.07")
regexes(r"""
\d+\.?\d*
\d*\.?\d*
\d*\.?\d{0,2}
\d+\.?\d{0,2}
""")
inputs("abc-de-1234, f-q-7, oh-no-33, coo-l-007")
regexes(r"""
[a-z]+-[a-z]{1,2}-\d+
[a-z]+-[a-z]{0,2}-\d+
[a-z]{1,3}-[a-z]{1,2}-\d+
""")
inputs("236.1, 8736.9999, 0.43, 72.875")
regexes(r"""
\d+\.\d+
\d*\.\d+
\d+\.\d*
""")
inputs("tw, *mcaaa, *qqee*, hello")
regexes(r"""
\*?[a-z]+\*?
\*?[a-z]+(e\*)?
\*?[a-z]{2,}\*?
\*?[a-z]+(e{2}\*)?
\*?[a-z]{2,}(e\*)?
""")
inputs("4567, +9752, 3015, +1")
regexes(r"""
\+?\d+
\+?\d*
\d?\+?\d+
""")
inputs("12.5, 67.5, 89.5, 93.5")
regexes(r"""
\d{2}\.5
\d{2}\.\d
\d+\.5
\d[0-9A-Za-z]\.5
[0-9A-Za-z]\d\.5
""")
inputs("C05678, C07123, C09223, C01241")
regexes(r"""
C0\d{4}
C0\d+
C\d{5}
C0\d{2,}
C\d+
""")
inputs("-54,89, 7, -902, 9,3")
regexes(r"""
-?\d{1,2},?\d*
-?\d*,?\d{1,2}
-?\d{0,2},?\d*
-?\d*,?\d{0,2}
-?\d*,?\d+
""")
inputs("B98734, g67321, A73021, k00883")
regexes(r"""
[A-Za-z]\d{5}
[0-9A-Za-z]\d{5}
[A-Za-z]\d+
[0-9A-Za-z]\d+
[A-Za-z]\d{2,}
""")
inputs("Hello Bob, Sunil Kumar, Jack Sparrow, Oh No")
regexes(r"""
[A-Z][a-z]+ [A-Z][a-z]+
[A-Z][a-z]+ [A-Za-z][a-z]+
[A-Za-z][a-z]+ [A-Z][a-z]+
[A-Z][a-z]+ [0-9A-Za-z][a-z]+
[0-9A-Za-z][a-z]+ [A-Z][a-z]+
""")
inputs("9123956745, 9876432129, 9333444521, 9438827087")
regexes(r"""
9\d+
9\d{2,}
\d+
9\d{3,}
\d{2,}
""")
inputs("H347gjdj, 8, sdjW23, Q3QW")
regexes(r"""
[0-9A-Za-z]+
[0-9A-Za-z]*
(H3)?[0-9A-Za-z]+
(Q3)?[0-9A-Za-z]+
""")
inputs("3, 890567, 345, 77")
regexes(r"""
\d+
\d*
""")
inputs("123 45, 890 32, 999 54, 764 51")
regexes(r"""
\d{3} \d{2}
\d+ \d{2}
\d{2,} \d{2}
\d{3} \d+
\d{3} \d[0-9A-Za-z]
""")


