#include <iostream>
#include <iomanip>
#include <cublas_v2.h>
#include "helper.cuh"
#include <cublasLt.h>
using namespace helper;

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}


int main()
{
    //indexing in c style
    int M, N, K;
    M = 16; //rows of weight matrix
    N = 2; //batch size
    K = 16; //column of weight matrix , rows of input matrix
    // A = M x K
    // B= K x N
    // C = M x N
    // These are standard math notations


    float alpha = 1;
    float beta = 0;

	// We define our matrices and initialize them with random variables
    my_data<int8_t> A, B;
    my_data<int8_t> C;

    A.init(M*K);
    // Below A is initiazed in a cublas-trasnformed fashin aka regular cpp form 
    // this means that memory is sequentially addresssed for both A and B
    // A is stored in row major fashion now as it is transposed
	// Use this initializer to make A identity matrix
    //A.init(K * M,true,K,M); 
    B.init(K*N);
    C.init(M*N);

    cudaMalloc(&A.d_ptr_, A.size_);
    cudaMalloc(&B.d_ptr_, B.size_);
    cudaMalloc(&C.d_ptr_, C.size_);

    
    cudaMemcpy(A.d_ptr_, A.h_ptr_, A.size_, cudaMemcpyHostToDevice);
    cudaMemcpy(B.d_ptr_, B.h_ptr_, B.size_, cudaMemcpyHostToDevice);
    cudaMemcpy(C.d_ptr_, C.h_ptr_, C.size_, cudaMemcpyHostToDevice);
    
    
    
    
    std::cout << "A(T):" << std::endl;
    // A matrix was transposed to need to take that into account when printing
    printMatrix(A.h_ptr_, M, K, true);
	
    std::cout << "B:" << std::endl;
    printMatrix(B.h_ptr_, K, N);
    


    void *workspace;
    size_t workspaceSize = 1024 * 1024 * 8;
    cudaMalloc(&workspace, workspaceSize);
    
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, K, M, K);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8I, M, N, M);

    cublasLtMatmulPreferenceCreate(&preference);
    
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    //cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
    std::cout<<"List of algos :"<<returnedResults<<std::endl;
	// create cuda event handles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //std::cout<<cublasLtGetVersion()<<std::endl;
    cudaEventRecord(start,0);
        
    cublasLtMatmul(ltHandle,
        operationDesc,
        &alpha,
        A.d_ptr_,
        Adesc,
        B.d_ptr_,
        Bdesc,
        &beta,
        C.d_ptr_,
        Cdesc,
        C.d_ptr_,
        Cdesc,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        0);


    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float cudaElapsedTime;
    cudaEventElapsedTime(&cudaElapsedTime, start, stop);
    cudaMemcpy(C.h_ptr_, C.d_ptr_, C.size_, cudaMemcpyDeviceToHost);

    std::cout << "C out:" << std::endl;
    printMatrix(C.h_ptr_, M, N);
    
    std::cout << std::setw(4) << cudaElapsedTime << " ms" << std::endl;
    cudaFree(A.d_ptr_);
    cudaFree(B.d_ptr_);
    cudaFree(C.d_ptr_);
	cublasLtDestroy(ltHandle);
    return 0;


}
