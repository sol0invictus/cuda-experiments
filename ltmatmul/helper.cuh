#include <cstdio>
#include <cfloat>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <cassert>
//Helper function to convert row major to column major
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

namespace helper{

    template <typename D>
    class my_data
    {
        
        public:
        int size_;
        D* h_ptr_;
        D* d_ptr_;
        my_data()
        {
            h_ptr_ = NULL;
            d_ptr_ = NULL;
            size_ = 0;
        }
        ~my_data()
        {
            if (h_ptr_ != NULL)
                delete[] h_ptr_;
            if (d_ptr_ != NULL)
                cudaFree(d_ptr_);
        }
        void init(int size, bool eye = false, int row = 1,int col = 1)
        {

            size_ = size * sizeof(D);
            h_ptr_ = (D *)new D[size_];
            d_ptr_ = NULL;
            //int ctr=0;
            for (int i = 0; i < size_; i++){
                    if (typeid(D) == typeid(float)) {
                        if (eye==false){
                            h_ptr_[i] = .1f * rand() / (float)RAND_MAX;
                        }
                        else{
                            int row_ = (int)i%row;
                            int col_ = (int)i/row;
                            if (row_==col_){
                                h_ptr_[i] = 1.f;
                            }
                            else
                                h_ptr_[i] = 0.f;
                        }

                        
                        //h_ptr_[i] = 1.f;
                    }
                    else{
                        if (eye==false){
                            h_ptr_[i] = rand();
                        //h_ptr_[i] = 1;
                        }
                        else{
                            int row_ = (int)i%row;
                            int col_ = (int)i/row;
                            if (row_==col_){
                                h_ptr_[i] = 1;
                            }
                            else
                                h_ptr_[i] = 0;
                        }
                    }
                        
                    
            }

            //std::cout<<ctr<<std::endl;

            
        }
    };

void printMatrix(const float *matrix, const int ldm, const int n, bool is_t = false) {
    for (int i = 0; i < ldm; i++) {
        for (int j = 0; j < n; j++) {
            if (is_t==false)
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
            else
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[i*n+j];
        }
        std::cout << std::endl;
    }
}

void printMatrix(const char *matrix, const int ldm, const int n, bool is_t = false) {
    for (int i = 0; i < ldm; i++) {
        for (int j = 0; j < n; j++) {
            if (is_t==false)
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << (int)matrix[IDX2C(i, j, ldm)];
            else
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << (int)matrix[i*n+j];
        }
        std::cout << std::endl;
    }
}

void printMatrix(const int *matrix, const int ldm, const int n, bool is_t = false) {
    for (int i = 0; i < ldm; i++) {
        for (int j = 0; j < n; j++) {
            if (is_t==false)
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
            else
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[i*n+j];
        }
        std::cout << std::endl;
    }
}

void printMatrix(const int8_t *matrix, const int ldm, const int n, bool is_t = false) {
    for (int i = 0; i < ldm; i++) {
        for (int j = 0; j < n; j++) {
            if (is_t==false)
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << (int)matrix[IDX2C(i, j, ldm)];
            else
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << (int)matrix[i*n+j];
        }
        std::cout << std::endl;
    }
}
}



