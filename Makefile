.PHONY: test

singlethread_cpu.s: singlethread_cpu.o Makefile
	objdump -dC -M intel singlethread_cpu.o > singlethread_cpu.s

singlethread_cpu.o: include/hcde.hh src/common.hh src/singlethread_cpu.cc Makefile
	clang++ src/singlethread_cpu.cc -Iinclude -O3 -march=native -mtune=native -DNDEBUG=1 -c -std=c++17 -osinglethread_cpu.o -ggdb

test_bin: src/singlethread_cpu.cc test/test.cc include/hcde.hh src/common.hh Makefile
	clang++ -otest_bin src/singlethread_cpu.cc test/test.cc -Iinclude -std=c++17 -ggdb -Wall -Wextra

compress: src/singlethread_cpu.cc src/compress.cc include/hcde.hh src/common.hh Makefile
	clang++ src/singlethread_cpu.cc src/compress.cc -Iinclude -O3 -march=native -mtune=native -std=c++17 -ocompress -lboost_program_options

test: test_bin Makefile
	./test_bin

