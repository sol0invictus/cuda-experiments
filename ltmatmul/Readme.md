Int8 matrix multiplication via Cublas-Lt
----------------------------------

This example showcases matrix multiplication via Cublas-LT library.
cublas-lt contains cublasltmatmul which supports int8 matrix multiplication with int8 output.
This library also supports both column major and row major memory indexing and is very performance efficient.
I recommend using this instead of cublasgemm.

Some undocumented facts:

1) Int8 is only supported on Cuda compute >= 7.5
2) Matrix dimensions for Int8 multiplication need to be multiples of 16 for int8 ouput.
3) Matrix dimensions for Int8 multiplication need to be multiples of 4 for fp32/int32 ouput.
