.PHONY: test

singlethread_cpu.s: singlethread_cpu.o
	objdump -dC -M intel singlethread_cpu.o > singlethread_cpu.s

singlethread_cpu.o: include/hcde.hh src/common.hh src/singlethread_cpu.cc
	clang++ src/singlethread_cpu.cc -Iinclude -O3 -march=native -mtune=native -DNDEBUG=1 -c -std=c++17 -osinglethread_cpu.o -ggdb

test_bin: test/test.cc include/hcde.hh src/common.hh
	g++ -otest_bin test/test.cc -Iinclude -std=c++17 -ggdb

test: test_bin
	./test_bin

