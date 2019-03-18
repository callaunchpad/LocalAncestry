class Peekable():

    def __init__(self, filename = None, file = None):
        assert (file is not None) is not (filename is not None), "Only one of filename and file must be defined"
        if filename is not None:
            self._file = open(filename, 'r')
        elif file is not None:
            self._file = file

        self.next_line = None

    def peek(self):
        if self.next_line is None:
            self.next_line = next(self._file)
        return self.next_line

    def __next__(self):
        if self.next_line is None:
            return next(self._file)
        else:
            temp = self.next_line
            self.next_line = None
            return temp

    def __iter__(self):
        return self