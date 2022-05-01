class Example:
    def __init__(self, *args):
        self.inputs = args
        self.regex = args[-1]

all_benchmarks = [
    Example(
        "t",
        "alex",
        "ramon",
        "bob"),
    Example(
        "123.12",
        "2",
        "6",
        "9.54"),
    Example(
        "091234567", "098764321", "093334445"),
    Example("{foo", "{bar", "{nice"),
    Example("a_bc.h", "a_bc.lol", "a_bc.net"),
    Example("123456", "789012", "666666"),
    Example("390987.456", "123456.789", "789012.543"),
    Example("5X^7", "44X^9", "X", "78X"),
    Example("bcDD", "hH", "tellMEE", "byeBYE"),
    Example("123456789.123", "96781.4", "5.6", "89325.72"),
    Example("5.7456", "9.23", "8.1", "7.9032"),
    Example("1234", "+9", "432", "8"),
    Example("1-2", "345-6732", "234444-395749"),
    Example("1566", "1566-87", "1566-00"),
    Example("3", "4.500000", "27.0000000000000", "5."),
    Example("Geeks", "camelCase", "39dih200-Ke"),
    Example("202-918-2132", "678-978-0459", "205-521-0797"),
    Example("a-b 0", "CdeF-gh 1234", "xYz-XP 588"),
    Example("1.0", "2.4683", "8825729.33"),
    Example("tw", "*mcaaa", "*qqee*"),
    Example("12.5", "34.5", "56.5"),
    Example("1", "1.0", "9876543210987", "123456789012345678.6"),
    Example("1357924", "0875288", "1957372493", "1000000000"),
    Example("C01234", "C05678", "C01113")]
