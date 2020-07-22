CXX = /usr/bin/clang++
CXXFLAGS += -Iinclude -std=c++17 -Wall -Wextra -Werror=return-type -Werror=init-self
OPTFLAGS += -O3 -DNDEBUG=1
DEBUGFLAGS += -ggdb

HEADERS = include/hcde.hh src/common.hh src/fast_profile.hh src/strong_profile.hh

.PHONY: test clean

all: singlethread_cpu.s test_bin compress

singlethread_cpu.s: singlethread_cpu.o Makefile
	objdump -dC -M intel singlethread_cpu.o > singlethread_cpu.s

singlethread_cpu.o: $(HEADERS) src/singlethread_cpu.cc Makefile
	$(CXX) -osinglethread_cpu.o -c $(CXXFLAGS) $(OPTFLAGS) $(DEBUGFLAGS) src/singlethread_cpu.cc 

test_bin: $(HEADERS) src/singlethread_cpu.cc test/test.cc Makefile
	$(CXX) -otest_bin $(CXXFLAGS) $(DEBUGFLAGS) src/singlethread_cpu.cc test/test.cc

compress: $(HEADERS) src/singlethread_cpu.cc src/compress.cc Makefile
	$(CXX) -ocompress $(CXXFLAGS) $(OPTFLAGS) src/singlethread_cpu.cc src/compress.cc -lboost_program_options

test: test_bin Makefile
	./test_bin

clean:
	rm -f *.s *.o test_bin compress

