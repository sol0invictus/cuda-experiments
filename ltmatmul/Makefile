CXX := nvcc
TARGET :=  ltmatmul

all: ltmatmul

ltmatmul: $(TARGET).cu
	$(CXX) $(TARGET).cu -o $(TARGET) -lcublas -lcublasLt

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
