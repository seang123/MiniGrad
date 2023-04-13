
CXX = g++
VER = -std=c++17
#LDFLAGS = $(VER) -static -Wall -Wconversion -Wpedantic -Wextra -O3 -msse4.1 # -O3 # -Werror
LDFLAGS = $(VER) -static -Wall -Wconversion -O3 -msse4.1 # -O3 # -Werror




main: main.o tensor.o substance.o ops.o operations.o nn.o
	$(CXX) $(LDFLAGS) -o main main.o Tensor.o Substance.o Ops.o operations.o nn.o 

tests: tests.o tensor.o substance.o ops.o operations.o nn.o
	$(CXX) $(LDFLAGS) -o test tests.o tensor.o substance.o ops.o operations.o nn.o


debug: main.o tensor.o substance.o ops.o operations.o nn.o
	$(CXX) $(LDFLAGS) -o main main.o Tensor.o Substance.o Ops.o operations.o nn.o 


main.o: main.cpp
	$(CXX) -c $(LDFLAGS) main.cpp

#tests: tests.o
#	g++ -o tests tests.o

tests.o: tests.cpp
	$(CXX) -c $(LDFLAGS) tests.cpp

#tensor: tensor.o
#	g++ -o tensor tensor.o

tensor.o: Tensor.cpp Tensor.h Substance.h
	$(CXX) -c $(LDFLAGS) Tensor.cpp


#substance: substance.o
#	g++ -o substance substance.o

substance.o: Substance.cpp Substance.h
	$(CXX) -c $(LDFLAGS) Substance.cpp


#ops: ops.o
#	g++ -o ops ops.o

ops.o: Ops.cpp Ops.h
	$(CXX) -c $(LDFLAGS) Ops.cpp

operations.o: Operations.cpp
	$(CXX) -c $(LDFLAGS) Operations.cpp

nn.o: nn.cpp nn.h
	$(CXX) -c $(LDFLAGS) nn.cpp