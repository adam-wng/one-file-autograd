#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <stdio.h>
// #include <stdint.h>
// #include <inttypes.h>

#include <bitset>
#include <chrono>

uint64_t state = 950706376;
float bias_p = 0.25;
float selu_s = 1.05070098735;
float selu_a = 1.67326324235;
float selu_as = selu_a * selu_s;

float normal(uint64_t p1, uint64_t p2, float mu, float sigma){
    // Aux sample to normal distribution
    float sum = 0;
    while (p1 != 0) {
        sum = sum + p1 % 2;
        p1 = p1 >> 1;
    }
    while (p2 != 0) {
        sum = sum + p2 % 2;
        p2 = p2 >> 1;
    }
    return mu + sigma * (16*((sum/64)-0.5));
}

uint64_t LCG_31(uint64_t prev_state){
    // Aux LCG
    uint64_t temp = (uint64_t)(prev_state)*950706376;
    return temp % 4294967295;
}

float get_next_normal(float mu, float sigma){
    uint64_t temp = LCG_31(state);
    state = LCG_31(temp);
    return normal(state, temp, mu, sigma);
}

template <typename T>
void lengthen(int old_len, T ** ptr){
    T * temp = new T [old_len*2];
    for (int i = 0; i < old_len ; i++){
        temp[i] = (*ptr)[i];
    }
    // temp[1] =3;
    delete [] (*ptr);
    *ptr = temp; 
}

template <typename T>
void ptr_add(T ** ptr, int *array_len, int * cur_len, T& new_val){
    std::cout<<"&";
    if (*array_len <= *cur_len){
        lengthen(*array_len, ptr);
        *array_len *= 2;
    }
    // std::cout<<*cur_len;
    *ptr[*cur_len] = new_val;
    *cur_len += 1;
}

template <typename T>
void ptr_add(T ** ptr, int *array_len, int * cur_len, T&& new_val){
    std::cout<<"&&";
    if (*array_len <= *cur_len){
        lengthen(*array_len, ptr);
        *array_len *= 2;
    }
    // std::cout<<*cur_len;
    *(*ptr+*cur_len) = new_val;
    // *ptr[*cur_len] = new_val;
    *cur_len += 1;
}

template <typename T>
class array{
    public:
    T* vals;
    int alloc_mem; // allocated mem
    int numel; // num elements

    array(){
        vals = new T [1];
        alloc_mem = 1;
        numel = 0;
    }

    array(T * vals_, int array_len_){
        vals = vals_;
        alloc_mem = array_len_;
        numel = 0;
    }

    array(T * vals_, int array_len_, int cur_len_){
        vals = vals_;
        alloc_mem = array_len_;
        numel = cur_len_;
    }
    
    array<T>& operator=(array<T> & itm){
        vals = new T [itm.alloc_mem];
        for (int i = 0; i < itm.numel ; i++){
            vals[i] = itm.vals[i];
        }
        alloc_mem = itm.alloc_mem;
        numel = itm.numel;
        return *this;
    }
    
    array<T>& operator=(array<T> && itm){
        vals = new T [itm.alloc_mem];
        for (int i = 0; i < itm.numel ; i++){
            vals[i] = itm.vals[i];
        }
        alloc_mem = itm.alloc_mem;
        numel = itm.numel;
        return *this;
    }

    ~array(){
        delete [] vals;
    }

    void add(T & new_val){
        // std::cout<<"lvalue";
        if (alloc_mem <= numel){
            lengthen(alloc_mem, &vals);
            alloc_mem *= 2;
        }
        *(vals + numel) = new_val;
        numel += 1;
        // ptr_add(&vals, &alloc_mem, &numel, new_val);
    }

    void add(T && new_val){
        // std::cout<<"rvalue";
        if (alloc_mem <= numel){
            lengthen(alloc_mem, &vals);
            alloc_mem *= 2;
        }
        *(vals + numel) = new_val;
        numel += 1;
    }

    void set(T & new_val, int idx){
        if (alloc_mem <= idx){
            throw std::invalid_argument("idx larger than alloc");
        }
        *(vals + idx) = new_val;
        numel = std::max(numel+1, idx);
    }

    T& operator[](int idx){
        if (idx >= numel){
            throw std::invalid_argument("Idx larger than numel");
        }
        return vals[idx];
    }

    void reset(int alloc){
        // Deletes vals and allocates new array
        delete [] vals;
        vals = new T [alloc];
        alloc_mem = alloc;
        numel = 0;
    }
};

class Matrix{
    public:
    float * buf;
    uint32_t dim1;
    uint32_t dim2;
    uint64_t numel = 0;
    // bool grad = false;
    // TODO (low): bool parity = true; contiguous may help 

    Matrix(){
        dim1 = 0;
        dim2 = 0;
        numel = 0;
        buf = new float [2];
        // this->zero();
    }

    Matrix(uint32_t dim1_, uint32_t dim2_){
        dim1 = dim1_;
        dim2 = dim2_;
        numel = (uint64_t)dim1*(uint64_t)dim2;
        buf = new float [(int)(numel+1)];
        zero();
    }

    Matrix(uint32_t dim1_){
        dim1 = dim1_;
        dim2 = 1;
        numel = (uint64_t)dim1;
        buf = new float [(int)numel+1];
        this->zero();
    }

    Matrix(uint32_t dim1_, uint32_t dim2_, float mu, float sigma){
        dim1 = (uint32_t) dim1_;
        dim2 = (uint32_t) dim2_;
        numel = (uint64_t)dim1*(uint64_t)dim2;
        buf = new float [(int)numel+1];

        float * m_t = buf;
        for (uint64_t i = 0; i < numel; i++){
            *m_t = get_next_normal(mu, sigma);
            m_t++;
        }
    }

    Matrix(uint32_t dim1_, float mu, float sigma){
        dim1 = (uint32_t) dim1_;
        dim2 = (uint32_t) 1;
        numel = (uint64_t)dim1;
        buf = new float [(int)numel+1];

        float * m_t = buf;
        for (uint64_t i = 0; i < numel; i++){
            *m_t = get_next_normal(mu, sigma);
            m_t++;
        }
    }

    // Copy
    Matrix(Matrix & test){
        dim1 = test.dim1;
        dim2 = test.dim2;
        numel = (uint64_t)dim1*(uint64_t)dim2;
        buf = new float [(int)numel+1];
        float * bufc = buf;
        float * tbuf = test.buf;
        // zero();
        for (uint64_t i = 0; i < numel; i++){
            *bufc = *tbuf;
            bufc++;
            tbuf++;
        }
    }

    // Assignment
    Matrix& operator=(Matrix &test){
        // std::cout<<"m&";
        delete [] buf;
        buf = new float [(int)test.numel+1];
        for (uint64_t i = 0; i < test.numel; i++){
            buf[i] = test.buf[i];
        }
        numel = test.numel;
        dim1 = test.dim1;
        dim2 = test.dim2;
        // std::cout<<"m&";
        return *this;
    }

    Matrix& operator=(Matrix &&test){
        // std::cout<<"m&&";
        delete [] buf;
        buf = new float [(int)test.numel+1];
        for (uint64_t i = 0; i < test.numel; i++){
            buf[i] = test.buf[i];
        }
        numel = test.numel;
        dim1 = test.dim1;
        dim2 = test.dim2;
        return *this;
    }


    ~Matrix(){
        delete[] buf;
    }


    void zero(){
        float * m_t = buf;
        for (uint64_t i = 0; i < numel; i++){
            *m_t = 0;
            m_t++;
        }
    }

    float* i_(uint32_t idx1, uint32_t idx2){
        if ((idx1 > dim1) | (idx2 > dim2)){
            std::cout<<std::endl;
            std::cout<<idx1<<" "<<dim1<<" ";
            std::cout<<idx2<<" "<<dim2<<" ";
            throw std::invalid_argument("Index out of bounds");
        }
        return buf+idx1*dim2+idx2;
        // return buf[idx1*dim2+idx2];
    }

    float* i_(uint32_t idx1){
        if (idx1 > dim1){
            throw std::invalid_argument("Index out of bounds");
        }
        return buf + idx1*dim2;
    }

    Matrix T(){
        Matrix tgt = Matrix(dim1, dim2);
        tgt.zero();
        float * t = tgt.buf;
        for (uint32_t i = 0; i < dim2; i++){
            for (uint32_t j = 0; j < dim1; j++){
                *t = *(this->i_(j,i));
                t++;
            }
        }
        tgt.dim1 = dim2;
        tgt.dim2 = dim1;
        tgt.numel = numel;
        return tgt;
    }

    void copy_row(Matrix & src, int src_row, int tgt_row){
        if (src.numel - (uint64_t)dim2*(uint64_t)src_row < dim2){
            throw std::invalid_argument("Not enough params");
        }
        float * start = src.buf + src_row*dim2;
        float * copy_pt = buf + tgt_row * dim2;
        for (uint32_t i = 0; i < dim2; i++){
            *copy_pt = *start;
            copy_pt ++;
            start ++;
        }
    }

    // void add_row(Matrix & src, int src_row, int tgt_row){
    //     if (src.numel - (uint64_t)dim2*(uint64_t)src_row < dim2){
    //         throw std::invalid_argument("Not enough params");
    //     }
    //     float * start = src.buf + src_row*dim2;
    //     float * copy_pt = buf + tgt_row * dim2;
    //     for (int i = 0; i < dim2; i++){
    //         *copy_pt += *start;
    //         copy_pt ++;
    //         start ++;
    //     }
    // }

    Matrix operator+(Matrix & obj){
        if ((dim1 != obj.dim1) | (dim2 != obj.dim2)){
            if ((dim2 == obj.dim2) && (obj.dim1 == 1)){ // Check if vector addition
                Matrix tgt = Matrix(dim1, dim2);

                float * tgt_t = tgt.buf;
                float * first_t = buf;
                float * second_t = obj.buf;
                
                for (uint32_t i = 0; i < dim1; i++){
                    for (uint32_t j = 0; j < dim2; j++) {
                        *tgt_t = *first_t + *second_t;
                        tgt_t++;
                        first_t++;
                        second_t++;
                    }
                    second_t = obj.buf;
                }
                return tgt;
            } else if ((dim1 == obj.dim1) && (obj.dim2 == 1)){
                Matrix tgt = Matrix(dim1, dim2);

                float * tgt_t = tgt.buf;
                float * first_t = buf;
                float * second_t = obj.buf;
                
                for (uint32_t i = 0; i < dim1; i++){
                    for (uint32_t j = 0; j < dim2; j++) {
                        *tgt_t = *first_t + *second_t;
                        tgt_t++;
                        first_t++;
                    }
                    second_t++;
                }
                return tgt;
            }
            std::cout<<std::endl<<"Matrix 1: "<<dim1<<", "<<dim2;
            std::cout<<std::endl<<"Matrix 2: "<<obj.dim1<<", "<<obj.dim2;
            throw std::invalid_argument("Addition matrix sizes do not match");
        }

        Matrix tgt = Matrix(dim1, dim2);

        float * tgt_t = tgt.buf;
        float * first_t = buf;
        float * second_t = obj.buf;

        for (uint64_t i = 0; i < numel; i++){
            *tgt_t = *first_t + *second_t;
            tgt_t++;
            first_t++;
            second_t++;
        }
        return tgt;
    }

    void operator+=(Matrix & obj){
        if ((dim1 != obj.dim1) | (dim2 != obj.dim2)){
            std::cout<<std::endl<<"obj1: "<<dim1<<" "<<dim2<<" obj2: "<<obj.dim1<<" "<<obj.dim2;
            throw std::invalid_argument("Matrix sizes do not match +=&");
        }

        float * first_t = buf;
        float * second_t = obj.buf;

        for (uint64_t i = 0; i < numel; i++){
            *first_t += *second_t;
            first_t++;
            second_t++;
        }
    }

    void operator+=(Matrix && obj){
        if ((dim1 != obj.dim1) | (dim2 != obj.dim2)){
            throw std::invalid_argument("Matrix sizes do not match +=&&");
        }

        float * first_t = buf;
        float * second_t = obj.buf;

        for (uint64_t i = 0; i < numel; i++){
            *first_t += *second_t;
            first_t++;
            second_t++;
        }
    }

    Matrix operator+(int scalar){

        Matrix tgt = Matrix(*this);

        float * tgt_t = tgt.buf;
        float * first_t = buf;

        for (uint64_t i = 0; i < numel; i++){
            *tgt_t = *first_t + scalar;
            tgt_t++;
            first_t++;
        }
        return tgt;
    }

    Matrix operator*(float scalar){
        Matrix tgt = Matrix(*this);

        float * tgt_t = tgt.buf;
        float * first_t = buf;

        for (uint64_t i = 0; i < numel; i++){
            *tgt_t = *first_t * scalar;
            tgt_t++;
            first_t++;
        }
        return tgt;
    }

    void operator+=(int scalar){
        float * first_t = buf;

        for (uint64_t i = 0; i < numel; i++){
            *first_t += scalar;
            first_t++;
        }
    }

    void operator*=(float scalar){
        float * first_t = buf;

        for (uint64_t i = 0; i < numel; i++){
            *first_t *= scalar;
            first_t++;
        }
    }

    Matrix operator*(Matrix  obj){
        if ((dim2 != obj.dim1)){
            std::cout<<dim2<<" "<<obj.dim1;
            throw std::invalid_argument("Matrix sizes do not match *");
        }

        Matrix tgt = Matrix(dim1, obj.dim2);
        tgt.zero();

        for (uint32_t i = 0; i < dim1; i++){
            for (uint32_t j = 0; j < obj.dim2; j++){
                for (uint32_t k = 0; k < dim2; k++){
                    tgt.buf[i*obj.dim2+j] += *i_(i,k) * *obj.i_(k, j);
                }
            }
        }
        return tgt;
    }

    Matrix operator&(Matrix & obj){
        if ((dim1 != obj.dim1) | (dim2 != obj.dim2)){
            throw std::invalid_argument("Matrix sizes do not match &");
        }

        Matrix tgt = Matrix(dim1, dim2);

        float * tgt_t = tgt.buf;
        float * first_t = buf;
        float * second_t = obj.buf;

        for (uint64_t i = 0; i < numel; i++){
            *tgt_t = *first_t * *second_t;
            tgt_t++;
            first_t++;
            second_t++;
        }
        return tgt;
    }

    void selu_(){
        float * it = buf;
        for (uint64_t i = 0; i< numel; i++){
            if (*it < 0){
                *it = selu_as *( exp(*it) -1);
            } else {
                *it = *it * selu_s;
            }
            it++;
        }
    }

    void relu_(){
        float * it = buf;
        for (uint64_t i = 0; i< numel; i++){
            if (*it < 0){
                *it = 0;
            }
            it++;
        }
    }

    Matrix relu(){
        Matrix temp = Matrix(dim1, dim2);
        temp.zero();
        float * it = buf;
        float * it_t = temp.buf;
        for (uint64_t i = 0; i< numel; i++){
            if (*it < 0){
                *it = 0;
            }
            it++;
        }
        return temp;
    }

    void print(){
        std::cout<<std::endl<<"[[" << *i_(0, 0);
        for (uint32_t j = 1; j < dim2; j++){
            std::cout << ", "<<*i_(0, j);
        }
        std::cout << "]";
        for (uint32_t i = 1; i < dim1; i++){
            std::cout << ", [" << *i_(i,0);
            for (uint32_t j = 1; j < dim2; j++){
                std::cout << ", "<< *i_(i, j);
            }
            std::cout << "]";
        }
        std::cout << "]"<<std::endl;
    }
};


std::ostream & operator <<(std::ostream &os, Matrix &m){
    std::cout<<std::endl<<"print:";
    return os << m.dim1 << " " << m.dim2 << " " << m.numel<<std::endl;
}


class Module{
    public:
    bool zero_passes;

    Module(){
        // throw std::invalid_argument("Module cons should never be called");
    }
    
    ~Module(){
        // throw std::invalid_argument("Module dest should never be called");
    }

    Matrix forward(Matrix & t){
        // Expects num_act
        return t;
    }
    
    void forward(Matrix & t, Matrix & t_){
    }

    void backward(){
    }

    void get_weight(){
        std::cout<<"module";
    }

    void zero_grad(){
    }

    void update(){
    }
};


class IdxMatrix{ 
    public:
    uint32_t * buf;
    uint32_t dim1 = 0;

    IdxMatrix(){
        buf = nullptr;
    }

    IdxMatrix(uint32_t dim1_){
        buf = new uint32_t [dim1_];
        dim1 = dim1_;

        uint32_t * m_t = buf;
        for (uint32_t i = 0; i < dim1; i++){
            *m_t = 0;
            m_t++;
        }
    }

    IdxMatrix(uint32_t dim1_, uint32_t * src_buf){
        buf = new uint32_t [dim1_+1];
        dim1 = dim1_;
        this->populate(src_buf);
    }

    IdxMatrix& operator=(IdxMatrix &test){
        delete [] buf;
        buf = new uint32_t [test.dim1+1];
        dim1 = test.dim1;
        this->populate(test.buf);
        return *this;
    }

    void populate(uint32_t * src_buf){
        for (uint32_t i = 0; i < dim1; i++){
            buf[i]  = src_buf[i];
        }
    }

    void print(){
        for (uint32_t i = 0; i < dim1; i++){
            std::cout<<buf[i]<<" ";
        }
        std::cout<<std::endl;
    }

    uint32_t i_(int idx){
        return *(buf+idx);
    }

    ~IdxMatrix(){
        if (dim1 != 0){
            delete[] buf;
        }
    }
};


class Embedding : public Module{
    public:
    Matrix embd;
    Matrix pos_emb;
    uint32_t num_w;
    uint32_t emb_size;
    uint32_t sen_len;

    Matrix grad_embd;
    Matrix grad_pos_emb;
    float num_updates;

    uint16_t opt; // 0 = sdg, 1 = Adam
    Matrix grad_embd_aux;
    Matrix grad_pos_emb_aux;

    bool cleared = true;

    Embedding(){
    }

    Embedding(uint32_t words, uint32_t emb, uint32_t sen){
        num_updates = 0;   

        embd = Matrix(words, emb, 0, sqrt(1-bias_p));
        grad_embd = Matrix(words, emb);
        grad_embd_aux = Matrix(words, emb);
        num_w = words;
        emb_size = emb;
        
        pos_emb = Matrix(sen, emb, 0, sqrt(bias_p));
        grad_pos_emb = Matrix(sen, emb);
        grad_pos_emb_aux = Matrix(sen, emb);
        sen_len = sen;
    }

    Matrix forward(IdxMatrix & idxm){
        if (idxm.dim1 != sen_len){
            throw std::invalid_argument("IdxMatrix and sen_len do not match");
        }

        Matrix temp(sen_len, emb_size);

        for (uint32_t i = 0; i < sen_len; i++){
            for (uint32_t j = 0; j < emb_size; j++){
                temp.buf[i*emb_size+j] = embd.buf[idxm.i_(i)*emb_size+j];
            }
        }

        temp += pos_emb;
        return temp;
    }

    void backward(Matrix grad, IdxMatrix & src){
        num_updates += 1;
        if (cleared){
            float* g_ = grad.buf;
            for (uint32_t i = 0; i < sen_len; i++){
                float *  t_ = grad_embd.buf + src.i_(i)*emb_size;
                for (uint32_t j = 0; j < emb_size; j++){
                    *t_ += *g_;
                    t_++;
                    g_++;
                }
            }
            grad_pos_emb += grad;
        } else {
            throw std::invalid_argument("multi-pass not implemented yet");
        }
    }

    void get_weight(){
        std::cout<<"embd: "<<embd.dim1;
    }

    void update(float lr, bool mode){
        if (mode) {
            embd += grad_embd * (lr / num_updates);
            pos_emb += grad_pos_emb * (lr / num_updates);
        } else {
            embd += grad_embd * lr;
            pos_emb += grad_pos_emb * lr;
        }
    }

    void zero_grad(){
        grad_embd.zero();
        grad_pos_emb.zero();
        num_updates = 0;
    }
};


class LeftLinearB: public Module{
    public:
    Matrix weight;

    Matrix grad_weight;
    Matrix grad_weight_aux;
    uint16_t opt; // 0 = sdg, 1 = Adam

    float num_updates;

    LeftLinearB(){
    }

    LeftLinearB(uint32_t in_size, uint32_t out_size){
        float mu = 0;
        float sigma = 1/((float)in_size);

        num_updates = 0;

        weight = Matrix(out_size, in_size, mu, sqrt(sigma*sqrt(1-bias_p)));
        
        grad_weight = Matrix(out_size, in_size);
        grad_weight_aux = Matrix(out_size, in_size);
    }

    Matrix forward(Matrix & input){
        return weight * input;
    }

    void backward(Matrix &prev_grad, Matrix &activation){
        num_updates += 1;
        grad_weight += prev_grad* (activation.T());
        Matrix t = (weight.T())*prev_grad;
        prev_grad = t;
    }

    void get_weight(){
        std::cout<<"left: "<<weight.dim1;
    }

    void update(float lr, bool mode){
        if (mode){
            weight += grad_weight * (lr / num_updates);
        } else {
            weight += grad_weight * lr;
        }
    }

    void zero_grad(){
        grad_weight.zero();
        num_updates = 0;
    }
};


class RightLinearB: public Module{
    public:
    Matrix weight;
    Matrix bias;

    Matrix grad_weight;
    Matrix grad_bias;
    uint16_t opt; // 0 = sdg, 1 = Adam
    Matrix grad_weight_aux;
    Matrix grad_bias_aux;
 
    bool cleared = true;

    float num_updates;

    RightLinearB(){
    }

    RightLinearB(uint32_t in_size, uint32_t out_size, uint32_t sen_size){
        float mu = 0;
        float sigma = 1/((float)in_size);

        num_updates = 0;

        weight = Matrix(in_size, out_size, mu, sqrt(sigma*sqrt(1-bias_p)));
        bias = Matrix(sen_size, out_size, mu, sqrt(bias_p));

        grad_weight = Matrix(in_size, out_size);
        grad_bias = Matrix(sen_size, out_size);
        grad_weight_aux = Matrix(in_size, out_size);
        grad_bias_aux = Matrix(sen_size, out_size);
    }

    Matrix forward(Matrix & input){
        return (input * weight) + bias;
    }

    void backward(Matrix &prev_grad, Matrix &activation){
        num_updates += 1;
        if (cleared){
            grad_weight += (activation.T()) * prev_grad;
            grad_bias += prev_grad;
            Matrix t = prev_grad * (weight.T());
            prev_grad = t;
        } else {
            throw std::invalid_argument("multi-pass not implemented yet");
        }
    }

    void get_weight(){
        std::cout<<"right: "<<weight.dim1;
    }

    void update(float lr, bool mode){
        if (mode){
            weight += grad_weight * (lr / num_updates);
            bias += grad_bias * (lr / num_updates);
        } else {
            weight += grad_weight * lr;
            bias += grad_bias * lr;

        }
    }

    void zero_grad(){
        grad_weight.zero();
        grad_bias.zero();
        num_updates = 0;
    }
};


class selu: public Module{
    public:

    selu(){
    }

    void forward(Matrix & input, Matrix & store_grad){
        if ((input.dim1!=store_grad.dim1)|(input.dim2 != store_grad.dim2)){
            throw std::invalid_argument("Selu inputs dont match");
        }
        float * t_ = input.buf;
        float * g_ = store_grad.buf;
        
        for (uint32_t i = 0; i < input.dim1; i++){
            for (uint32_t j = 0; j < input.dim2; j++)
            {
                if (*t_ < 0)
                {
                    *t_ = selu_as *( exp(*t_) -1);
                    *g_ = *t_ + selu_as;

                } else {
                    *t_ = selu_s* *t_;
                    *g_ = selu_s;
                }
                t_++;
                g_++;
            }
        }
    }

    void backward(Matrix &prev_grad, Matrix &activation){
        Matrix t = prev_grad & activation;
        prev_grad = t;
    }

    void get_weight(){
        std::cout<<"selu";
    }
};

const float res_norm = 0.70710678;
class residual: public Module{
    public:
    residual(){
    }

    Matrix forward(Matrix & input1, Matrix & input2){
        return (input1 + input2)*res_norm;
    }

    void backward(Matrix & prev_grad, Matrix & grad){
        prev_grad *= res_norm;
        grad = prev_grad;
    }
};


class concat: public Module{
    public:
    int dim;
    concat(int dim_){
        if ((dim_!=0)&(dim_!=1)){
            throw std::invalid_argument("Dim not (yet) supported");
        }
        dim = dim_;
    }

    Matrix forward(Matrix & input1, Matrix &input2){
        Matrix m;
        if (dim == 0){
            if (input1.dim2!=input2.dim2){
                throw std::invalid_argument("Concat dims incorrect");
            }
            m = Matrix(input1.dim1+input2.dim1,input1.dim2);
            float * t_ = m.buf;
            float * c1 = input1.buf;
            for (uint32_t i = 0; i < input1.dim1; i++){
                for (uint32_t j = 0; j < input1.dim2; j++){
                    * t_ = *c1;
                    t_++;
                    c1++;
                }
            }
            c1 = input2.buf;
            for (uint32_t i = 0; i < input2.dim1; i++){
                for (uint32_t j = 0; j < input2.dim2; j++){
                    * t_ = *c1;
                    t_++;
                    c1++;
                }
            }
        } else if (dim == 1){
            if (input1.dim1!= input2.dim1){
                throw std::invalid_argument("Concat dims incorrect");
            }
            //add like x x
            m = Matrix(input1.dim1, input1.dim2+input2.dim2);
            float * t_ = m.buf;
            float * c1 = input1.buf;
            float * c2 = input2.buf;
            for (uint32_t i = 0; i < input1.dim1; i++){
                for (uint32_t j = 0; j < input1.dim2; j++){
                    *t_ = *c1;
                    c1 ++;
                    t_++;
                }
                for (uint32_t j = 0; j < input2.dim2; j++){
                    *t_ = *c2;
                    c2 ++;
                    t_++;
                }
            }
        } else{
            throw std::invalid_argument("Dim not (yet) supported");
        }
        // m.print();
        return m;
    }

    void backward(Matrix &prev_grad, uint32_t i1_size, Matrix& grad){
        // grads.reset(2);
        if (dim == 0){
            // vertical
            float * t = prev_grad.buf;
            // std::cout<<"backwards: "<<prev_grad.dim1<<" "<<prev_grad.dim2<<" ";
            // grads.add(Matrix(i1_size, prev_grad.dim2));

            Matrix temp (i1_size, prev_grad.dim2);
            float * c1 = temp.buf;
            for (uint32_t i = 0; i < i1_size; i++){
                for (uint32_t j = 0; j < prev_grad.dim2; j++){
                    * c1 = * t;
                    c1++;
                    t++;
                }
            }
            grad.dim1 = prev_grad.dim1-i1_size;
            grad.dim2 = prev_grad.dim2;
            grad.numel = grad.dim1*grad.dim2;
            delete [] grad.buf;
            grad.buf = new float [(int)grad.numel+1];

            c1 = grad.buf;
            
            for (uint32_t i = 0; i < prev_grad.dim1-i1_size; i++){
                for (uint32_t j = 0; j < prev_grad.dim2; j++){
                    * c1 = * t;
                    c1++;
                    t++;
                }
            }
            prev_grad = temp;
        } else {
            throw std::invalid_argument("dim not implemented");
        }
    }

    void backward(Matrix &prev_grad, uint32_t i1_size){
        if (dim == 0){
            // vertical
            float * t = prev_grad.buf;
            Matrix temp (i1_size, prev_grad.dim2);
            float * c1 = temp.buf;
            for (uint32_t i = 0; i < i1_size; i++){
                for (uint32_t j = 0; j < prev_grad.dim2; j++){
                    * c1 = * t;
                    c1++;
                    t++;
                }
            }
            prev_grad = temp;
        } else {
            throw std::invalid_argument("dim not implemented");
        }
    }
};


class CrossEntropy: public Module{
    public:

    CrossEntropy(){
    }

    void forward(Matrix & input, int idx){
        if ((input.dim1 != 1)){
            throw std::invalid_argument("input dim1 wrong");
        }
        float * t = input.buf;
        float total = 0;
        for (uint32_t i = 0; i < input.dim1; i++){
            for (uint32_t j = 0; j < input.dim2; j++){
                *t = exp(*t);
                total += *t;
                t++;
            }
        }
        total = -1/total;
        t = input.buf;
        for (uint32_t i = 0; i < input.dim1; i++){
            for (uint32_t j = 0; j < input.dim2; j++){
                *t = (*t)*total;
                t++;
            }
        }
        *(input.i_(0,idx)) += 1;
    }

    float forward_(Matrix & input, int idx){
        return log(-1*(*input.i_(0,idx)-1));
        
    }

    void backward(Matrix& grads, Matrix &activation){
        grads = activation;
    }

    void get_weight(){
        std::cout<<"entropy";
    }
};

class Average: public Module{
    public:

    Average(){
    }

};

int num_mod_type = 8;
class GradBlock{
    public: 

    // Module * mod;
    array<bool> mod_type;
    Embedding * Embedding_;
    LeftLinearB * LeftLinearB_;
    RightLinearB * RightLinearB_;
    selu * selu_;
    residual * residual_;
    concat * concat_;
    CrossEntropy * CrossEntropy_;
    Average * Average_;

    IdxMatrix * root;
    int root_idx;
    bool root_ = false;

    int tgt_idx;
    bool loss_node = false; 
    
    // list of input gradblock idx
    array<int> prev;

    // list of output gradblock idx
    array<int> next;

    // whether input includes a non-grad outside element
    bool outside_input = false;
    int outside_idx;

    // whether you need to store the output for later usage
    bool store_out = false;
    Matrix stored_out;

    // Whether to store incoming gradients
    bool store_grad = false;
    Matrix stored_grad;
    
    // Whether to use immediate incoming grad
    bool use_imm_grad = false;
    
    // Whether one needs to send outgoing gradients
    bool out_grad = false;
    // Whether one needs to send outgoing gradients
    array<bool> out_grad_;
    bool swap;

    Matrix activation;
    int concat_dim;

    GradBlock(){ 
    }

    GradBlock& operator=(GradBlock & itm){
        std::cout<<"c";
        std::cout<<itm.mod_type.numel;
        if (itm.mod_type[0]){
            Embedding_ = itm.Embedding_;
        } else if (itm.mod_type[1]) {
            LeftLinearB_ = itm.LeftLinearB_;
        } else if (itm.mod_type[2]) {
            RightLinearB_ = itm.RightLinearB_;
        } else if (itm.mod_type[3]) {
            selu_ = itm.selu_;
        } else if (itm.mod_type[4]) {
            residual_ = itm.residual_;
        } else if (itm.mod_type[5]) {
            concat_ = itm.concat_;
        } else if (itm.mod_type[6]){
            CrossEntropy_ = itm.CrossEntropy_;
        } else if (itm.mod_type[7]){
            Average_ = itm.Average_;
        } else {
            throw std::invalid_argument("incorrect");
        }
        root_ = itm.root_;
        if (root_){
            root_idx = itm.root_idx;
            root = itm.root;
        }
        loss_node = itm.loss_node;
        if (loss_node){
            tgt_idx = itm.tgt_idx;
        }
        // prev = itm.prev;
        // next = itm.next;
        store_out = itm.store_out;
        if (store_out){
            stored_out = itm.stored_out;
        }
        std::cout<<"c";
        return *this;
    }

    GradBlock& operator=(GradBlock && itm){ 
        std::cout<<"c";
        if (itm.mod_type[0]){
            Embedding_ = itm.Embedding_;
        } else if (itm.mod_type[1]) {
            LeftLinearB_ = itm.LeftLinearB_;
        } else if (itm.mod_type[2]) {
            RightLinearB_ = itm.RightLinearB_;
        } else if (itm.mod_type[3]) {
            selu_ = itm.selu_;
        } else if (itm.mod_type[4]) {
            residual_ = itm.residual_;
        } else if (itm.mod_type[5]) {
            concat_ = itm.concat_;
        } else if (itm.mod_type[6]){
            CrossEntropy_ = itm.CrossEntropy_;
        } else if (itm.mod_type[7]){
            Average_ = itm.Average_;
        } else {
            throw std::invalid_argument("incorrect");
        }
        std::cout<<"c";
        // mod = itm.mod;
        activation = itm.activation;
        root_ = itm.root_;
        if (root_){
            root_idx = itm.root_idx;
            root = itm.root;
        }
        loss_node = itm.loss_node;
        if (loss_node){
            tgt_idx = itm.tgt_idx;
        }
        prev = itm.prev;
        next = itm.next;
        store_out = itm.store_out;
        if (store_out){
            stored_out = itm.stored_out;
        }
        std::cout<<"copied";
        return *this;
    }
    
    GradBlock(Embedding * mod_){
        Embedding_ = mod_;
        // mod_type = array(new bool [7] {true, false, false, false, false, false, false}, 7, 7);
        mod_type = array(new bool [num_mod_type], num_mod_type);
        for (int i = 0; i < num_mod_type; i++){
            mod_type.add(false);
        }
        mod_type.vals[0] = true;
        prev = array(new int [10], 10, 0);
        next = array(new int [10], 10, 0);
        out_grad_ = array(new bool [1], 1, 0);
    }
    GradBlock(LeftLinearB * mod_){
        LeftLinearB_ = mod_;
        mod_type = array(new bool [num_mod_type], num_mod_type);
        for (int i = 0; i < num_mod_type; i++){
            mod_type.add(false);
        }
        mod_type.vals[1] = true;
        prev = array(new int [10], 10, 0);
        next = array(new int [10], 10, 0);
        out_grad_ = array(new bool [1], 1, 0);
    }
    GradBlock(RightLinearB * mod_){
        RightLinearB_ = mod_;
        mod_type = array(new bool [num_mod_type], num_mod_type);
        for (int i = 0; i < num_mod_type; i++){
            mod_type.add(false);
        }
        mod_type.vals[2] = true;
        prev = array(new int [10], 10, 0);
        next = array(new int [10], 10, 0);
        out_grad_ = array(new bool [1], 1, 0);
    }
    GradBlock(selu * mod_){
        selu_ = mod_;
        mod_type = array(new bool [num_mod_type], num_mod_type);
        for (int i = 0; i < num_mod_type; i++){
            mod_type.add(false);
        }
        mod_type.vals[3] = true;
        prev = array(new int [10], 10, 0);
        next = array(new int [10], 10, 0);
        out_grad_ = array(new bool [1], 1, 0);
    }
    GradBlock(residual * mod_){
        residual_ = mod_;
        mod_type = array(new bool [num_mod_type], num_mod_type);
        for (int i = 0; i < num_mod_type; i++){
            mod_type.add(false);
        }
        mod_type.vals[4] = true;
        prev = array(new int [10], 10, 0);
        next = array(new int [10], 10, 0);
        out_grad_ = array(new bool [2], 2, 0);
    }
    GradBlock(concat * mod_){
        concat_ = mod_;
        mod_type = array(new bool [num_mod_type], num_mod_type);
        for (int i = 0; i < num_mod_type; i++){
            mod_type.add(false);
        }
        mod_type.vals[5] = true;
        prev = array(new int [10], 10, 0);
        next = array(new int [10], 10, 0);
        out_grad_ = array(new bool [2], 2, 0);
    }
    GradBlock(CrossEntropy * mod_){
        CrossEntropy_ = mod_;
        mod_type = array(new bool [num_mod_type], num_mod_type);
        for (int i = 0; i < num_mod_type; i++){
            mod_type.add(false);
        }
        mod_type.vals[6] = true;
        prev = array(new int [10], 10, 0);
        next = array(new int [10], 10, 0);
        out_grad_ = array(new bool [1], 1, 0);
    }
    GradBlock(Average * mod_){
        Average_ = mod_;
        mod_type = array(new bool [num_mod_type], num_mod_type);
        for (int i = 0; i < num_mod_type; i++){
            mod_type.add(false);
        }
        mod_type.vals[7] = true;
        prev = array(new int [10], 10, 0);
        next = array(new int [10], 10, 0);
        out_grad_ = array(new bool [1], 1, 0);
    }

    ~GradBlock(){
    }

    void add_prev(int idx){
        prev.add(idx);
    }

    void add_next(int idx){
        next.add(idx);
    }

    void save_output(){
        store_out = true;
    }

    void loss_cal(){
        loss_node = true;
    }

    void forward(IdxMatrix & input, array<Matrix> & mem){
        if (mod_type[0]){
            mem.reset(1);
            mem.add(Embedding_->forward(input));
            root = &input;
            if (store_out){
                stored_out = mem[0];
            }
        } else {
            throw std::invalid_argument("incorrect input");
        }
    }

    void forward(array<Matrix> & mem){// TODO (Mid)
        if (mod_type[1]){
            activation = mem[0];
            Matrix t = LeftLinearB_->forward(mem[0]);
            mem.reset(1);
            mem.add(t);
            if (store_out){
                stored_out = mem[0];
            }
        } else if (mod_type[2]){
            activation = mem[0];
            Matrix t = RightLinearB_->forward(mem[0]);
            mem.reset(1);
            mem.add(t);
            if (store_out){
                stored_out = mem[0];
            }
        } else if (mod_type[3]){
            activation = mem[0];
            selu_->forward(mem[0],activation);
            if (store_out){
                stored_out = mem[0];
            }
        } else if (mod_type[4]){
            Matrix t = residual_->forward(mem[0], mem[1]);
            mem.reset(1);
            mem.add(t);
            if (store_out){
                stored_out = mem[0];
            }
        } else if (mod_type[5]){
            Matrix t = concat_->forward(mem[0], mem[1]);
            if (concat_->dim == 0){
                concat_dim = mem[0].dim1;
            } else if (concat_->dim == 1){
                concat_dim = mem[0].dim2;
            }
            mem.reset(1);
            mem.add(t);
            if (store_out){
                stored_out = mem[0];
            }
        } else {
            throw std::invalid_argument("incorrect input");
        }
    }

    void forward(array<Matrix> & mem, int idx){
        if (mod_type[6]){
            CrossEntropy_->forward(mem[0], idx);
            activation = mem[0];
        } else {
            throw std::invalid_argument("incorrect input");
        }
    }

    void backward(Matrix & grad, float norm){
        if (mod_type[6]){// loss node
            CrossEntropy_->backward(grad, activation);
            grad *= norm;
        }else {
            throw std::invalid_argument("incorrect input");
        }
    }
    void backward(Matrix & grad){
        if (mod_type[0]){ // embd
            Embedding_->backward(grad, *root);
        } else if (mod_type[1]){ // left
            LeftLinearB_->backward(grad, activation);
        } else if (mod_type[2]){ // right
            RightLinearB_->backward(grad, activation);
        } else if (mod_type[3]){ // selu
            selu_->backward(grad, activation);
        } else if (mod_type[5]){// concat + outside node
            concat_->backward(grad, concat_dim);
        // } else if (mod_type[6]){// loss node
        //     CrossEntropy_->backward(grad, activation);
        // } else {
        } else {
            throw std::invalid_argument("incorrect input");
        }
    }

    void backward(Matrix & grad, Matrix & other_grad){
        if (mod_type[4]){//res
            residual_->backward(grad, other_grad);
        } else if (mod_type[5]){//concat
            concat_->backward(grad, concat_dim, other_grad);
        } else {
            throw std::invalid_argument("incorrect input");
        }
    }
};

bool check_membership(array<int> & list, int check){
    for (int i = 0; i < list.numel; i++){
        if (list[i] == check){
            return true;
        }
    }
    return false;
}

class Tree{
    // Class to construct
    // Essentially, this is going to be a list of unique GradBlocks, where each GradBlock stores a fn (not unique), its sources, and targets
    // First, one manually specifies the model using step and save_var. Then the model is run simply by iterating through the list
    public:
    array<GradBlock*> module_hist;
    // GradArray module_hist = GradArray();
    // GradArray module_hist;

    array<int> forward_order;
    array<int> reverse_order;
    bool compiled = false;

    // For rnn // TODO (Mid): MAKE SURE NOT DOUBLE COUNTING WITH INFO IN GRADBLOCK
    array<int> inputs;
    array<int> outputs;

    Tree(){
        module_hist = array(new GradBlock*[10], 10, 0);
        // module_hist = GradArray();
        inputs = array(new int[10], 10, 0);
        outputs = array(new int[10], 10, 0);
    }

    ~Tree(){ // TODO Mid
    }

    int check_prev(GradBlock * itm, int num_visited, int * visited){
        int cons;
        bool found = false;
        for (int i = 0; i< itm->prev.numel; i++){
            cons = itm->prev[i];
            for (int j = 0; j < num_visited; j++){
                if (visited[j] == cons){
                    found = true;
                    break;
                }
            }
            if (!found){
                return cons;
            }
        }
        return num_visited;
    }

    template <typename T>
    int add(T * current, int root){
    // int add(Module * current, int root){
        // For inputs

        // Check if the list of past modules needs to lengthened
        // module_hist.add(current);
        GradBlock *temp = new GradBlock(current);
        module_hist.add(temp);

        // Since this is an embedding, add this to the inputs list
        // module_hist[module_hist.numel-1].root_idx = inputs.numel;
        // module_hist[module_hist.numel-1].root_ = true;
        module_hist[module_hist.numel-1]->root_idx = inputs.numel;
        module_hist[module_hist.numel-1]->root_ = true;

        inputs.add(module_hist.numel - 1);

        return module_hist.numel - 1;
    }
    template <typename T>
    int add(T * current, int inside, int outside){

        GradBlock *temp = new GradBlock(current);
        module_hist.add(temp);

        module_hist[module_hist.numel-1]->add_prev(inside);
        module_hist[inside]->add_next(module_hist.numel-1);

        module_hist[module_hist.numel-1]->outside_idx = outside;
        module_hist[module_hist.numel-1]->outside_input = true; 

        return module_hist.numel - 1;
    }

    void print(){
        std::cout<<std::endl<<std::endl<<"Printing relations:"<<std::endl;
        for (int i = 0; i < module_hist.numel; i++){
            std::cout << "mod type:";
            for (int j = 0; j < module_hist[i]->mod_type.numel; j++){
                std::cout << module_hist[i]->mod_type[j]<<", ";
            }

            std::cout << std::endl << "mod idx" << i << " prev: " << std::endl;
            for (int j = 0; j < module_hist[i]->prev.numel; j++){
                std::cout << module_hist[i]->prev[j]<<", ";
            }
            std::cout << std::endl << "mod idx" << i << " next: " << std::endl;
            for (int j = 0; j < module_hist[i]->next.numel; j++){
                std::cout << module_hist[i]->next[j]<<", ";
            }
            std::cout<<std::endl;
        }
    }

    template <typename T>
    int add(T * current, int prev, bool loss_node){

        GradBlock * temp = new GradBlock(current);
        module_hist.add(temp);

        module_hist[module_hist.numel-1]->add_prev(prev);
        module_hist[prev]->add_next(module_hist.numel-1);

        if (loss_node) {
            module_hist[module_hist.numel-1]->loss_node = true; 
            module_hist[module_hist.numel-1]->tgt_idx = outputs.numel; 
            outputs.add(module_hist.numel-1);
        }
        return module_hist.numel - 1;
    }

    template <typename T>
    int add(array<T> * current, int location, int prev, bool loss_node){

        GradBlock * temp = new GradBlock(current);
        module_hist.add(temp);

        module_hist[module_hist.numel-1]->add_prev(prev);
        module_hist[prev]->add_next(module_hist.numel-1);

        if (loss_node) {
            module_hist[module_hist.numel-1]->loss_cal();
        }
        return module_hist.numel - 1;
    }

    template <typename T>
    int add(T * current, array<int> * prev, bool loss_node){
        // For all other inputs

        GradBlock * temp = new GradBlock(current);
        module_hist.add(temp);
        for (int i = 0; i < prev->numel; i++){
            module_hist[module_hist.numel-1]->add_prev((*prev)[i]);
            module_hist[(*prev)[i]]->add_next(module_hist.numel-1);
        }

        if (loss_node) {
            module_hist[module_hist.numel-1]->loss_cal(); 
        }

        return module_hist.numel - 1;
    }

    int find_new_root_forward(array<int> & visited, int cons_idx){
        int cons_prev;
        // bool computed;
        bool all_computed = true;
        
        // Iterate through the previous elements
        for (int i = 0; i < module_hist[cons_idx]->prev.numel; i++){
            cons_prev = module_hist[cons_idx]->prev[i];

            // Check if computed before 
            if (!check_membership(visited, cons_prev)){
                all_computed = false;
                break;
            }
        }
        if (all_computed){
            return cons_idx;
        } else {
            return find_new_root_forward(visited, cons_prev);
        }
    }

    int find_new_root_backward(array<int> & visited, int cons_idx){
        int cons_next;
        bool all_computed = true;
        
        // Iterate through the previous elements
        for (int i = 0; i < module_hist[cons_idx]->next.numel; i++){
            cons_next = module_hist[cons_idx]->next[i];

            // Check if computed before 
            if (!check_membership(visited, cons_next)){
                all_computed = false;
                break;
            }
        }

        if (all_computed){
            return cons_idx;
        } else {
            return find_new_root_backward(visited, cons_next);
        }
    }

    void compile(){
        // First pass through
        // Assumes the last element of module_hist to be the one that aggregates loss

        forward_order = array(new int[10], 10, 0);

        bool not_finished = true;
        bool membership;
        int last_module = 0;
        int current_module = 0;
        int next_compute;

        GradBlock * current_mod;
        Matrix current_matrix;
        Matrix * current_matrix_list;

        // check if assumptions are met
        int num_av = 0;
        for (int i = 0; i < module_hist.numel; i++){
            if (module_hist[i]->next.numel>2){
                throw std::invalid_argument("Invalid arch: outputs can only be used at most twice");
            }
            if (module_hist[i]->next.numel==0){
                for (int j = 0; j < module_hist[i]->prev.numel; j++){
                    if (!module_hist[module_hist[i]->prev[j]]->loss_node){
                        throw std::invalid_argument("Invalid arch: only loss nodes can be aggregated into final node");
                    }
                }
                num_av+=1;
            }
        }
        if (num_av != 1){
            throw std::invalid_argument("Invalid arch: only single final node permitted");
        }

        // forward_pass
        while (not_finished){
            current_mod = module_hist.vals[current_module];
            
            if (current_mod->prev.numel != 0){
                next_compute = find_new_root_forward(forward_order, current_module); 

                if (next_compute != current_module){
                    module_hist[last_module]->save_output();
                    current_module = next_compute;
                    current_mod = module_hist.vals[current_module];
                    if (current_mod->prev.numel != 0){
                        for (int i = 0; i < current_mod->prev.numel; i++){
                            module_hist[current_mod->prev[i]]->save_output();
                        }
                    }
                } else {
                    for (int i = 0; i < current_mod->prev.numel; i++){
                        if (current_mod->prev[i] != last_module){
                            module_hist[current_mod->prev[i]]->save_output();
                        }
                    }
                }
            }
            
            last_module = current_module;

            forward_order.add(current_module);

            // Find next item, since we are assuming that the only end node is the output
            for (int i = 0; i < current_mod->next.numel; i++){
                membership = check_membership(forward_order, current_mod->next[i]);
                if (!membership){
                    current_module = current_mod->next[i];
                    break;
                }
            }

            // if we are at the final node, check to see if we are able to compute it
            if (current_mod->next.numel == 0){
                next_compute = find_new_root_forward(forward_order, current_module); 

                // If we can, then finish
                if (next_compute == current_module){
                    // forward_order.add(current_module);
                    // current_mod->save_output();
                    not_finished = false;
                    break;
                } else{
                    // Ensure we can calculate this next node before proceeding
                    module_hist[last_module]->save_output();
                    if (current_mod->prev.numel != 0){ // whatever, readability
                        for (int i = 0; i < current_mod->prev.numel; i++){
                            module_hist[current_mod->prev[i]]->save_output();
                        }
                    }
                    current_module = next_compute;
                }
            }
        }
        
        reverse_order = array(new int[10], 10, 0);
        // Notice previous construction ensures that current_module is the final node

        not_finished = true;
        // Build backward
        while (not_finished){
            current_mod = module_hist.vals[current_module];
            if (current_mod->next.numel != 0){
                // check and make sure that all of the prev modules have been computed
                current_module = find_new_root_backward(reverse_order, current_module);
            }
            current_mod = module_hist.vals[current_module];


            // if all is set add the module idx
            reverse_order.add(current_module);
            
            // see if we are at a root node
            if (current_mod->prev.numel == 0){
                for (int i = 0; i < inputs.numel; i++){
                    not_finished = not_finished && check_membership(reverse_order, inputs[i]);
                    if (!not_finished){ // if found element not in list yet
                        current_module = find_new_root_backward(reverse_order, inputs[i]);
                        break;
                    }
                }
                // regenerate or break
                not_finished = !not_finished;
            } else {
                for (int i = current_mod->prev.numel-1; i >= 0; i--){
                    if (check_membership(reverse_order, current_mod->prev[i])){
                        throw std::invalid_argument("impossible grad tree");
                    } else {
                        current_module = current_mod->prev[i];
                        break;
                    }
                }
            }
        }

        // Once backwards order is computed, set up the gradient order
        for (int i = 0; i < reverse_order.numel; i++){
            // Iterate through the reverse order

            // Get idx
            current_module = reverse_order[i];
            current_mod = module_hist.vals[current_module];

            for (int j = 0; j < current_mod->prev.numel; j++){
                // for all previous items
                if (current_mod->prev[j] != reverse_order[i+1]){
                    // if the current input is not the next one to be computed
                    current_mod->out_grad = true;
                    current_mod->out_grad_.add(true);
                    module_hist.vals[current_mod->prev[j]]->store_grad = true;
                } else {
                    current_mod->out_grad_.add(false);
                    if (j!=0){
                        current_mod->swap=true;
                    } else {
                        current_mod->swap=false;
                    }
                }
            }
            for (int j = 0; j < current_mod->next.numel; j++){
                if (current_mod->next[j] == reverse_order[i-1]){
                    current_mod->use_imm_grad = true;
                }
            }

            if (current_mod->next.numel > 1 & i != 0){
                bool direct = false;

                for (int j = 0; j < current_mod->next.numel; j++){
                    direct = direct | (current_mod->next[j] == reverse_order[i-1]);
                }
                if (!direct){
                    std::cout<<std::endl<<i<<" ";
                    // throw std::invalid_argument("Arch compile error");
                }
            }
        }
        compiled = true;
    }

    void check_forward_backward(){
        array<int> forw_missing(new int [10],10);
        array<int> back_missing(new int [10],10);
        for (int i = 0; i < module_hist.numel; i++){
            if (!check_membership(forward_order, i)){
                forw_missing.add(i);
            }
            if (!check_membership(reverse_order, i)){
                back_missing.add(i);
            }
        }
        std::cout<<"Forward missing: ";
        for (int i = 0; i < forw_missing.numel; i++)
        {
            std::cout<< forw_missing[i] <<", ";
        }
        std::cout<<std::endl<<"Backwards missing: ";
        for (int i = 0; i < back_missing.numel; i++)
        {
            std::cout<< back_missing[i] <<", ";
        }
        std::cout<<std::endl<<"F:"<<forward_order.numel<<" B: "<<reverse_order.numel<<" T: "<<module_hist.numel<<std::endl;
    }

    void forward(array<IdxMatrix>& input, array<int>& tgt, Matrix& recurr_init){
        // Takes in an array of inputs
        // Assumes the last element of module_hist to be the one that aggregates loss
        if (input.numel != inputs.numel){
            throw std::invalid_argument("Incompatible number of inputs");
        }
        if (!compiled){
            throw std::invalid_argument("Must compile");
        }
        GradBlock * current_mod; 
        array<Matrix> mem = array(new Matrix [1], 1, 0);
        for (int i = 0; i< forward_order.numel; i++){
            current_mod = module_hist.vals[forward_order[i]];
            if (current_mod->root_){
                // for (int k = 0; k < input[(current_mod->root_idx)].dim1; k++){
                //     std::cout<<input[(current_mod->root_idx)].buf[k]<<" ";
                // }
                current_mod->forward(input[(current_mod->root_idx)], mem);

            } else if (current_mod->loss_node){

                current_mod->forward(mem, tgt[current_mod->tgt_idx]);

            } else if (current_mod->next.numel == 0){
                // // TODO REMOVE TESTING CODE
                std::cout<<"average";
                float sum = 0;
                float temp_val;
                for (int j = 0; j < current_mod->prev.numel; j++){
                    std::cout<<j<<" ";
                    GradBlock * temp_gb = module_hist.vals[current_mod->prev[j]];
                    temp_val = temp_gb->CrossEntropy_->forward_(temp_gb->activation, tgt[temp_gb->tgt_idx]);
                    std::printf("%.16f", temp_val);
                    // std::cout<< temp_val;
                    std::cout<<" ";
                    sum += temp_val;
                }
                std::cout<<std::endl<<"total: ";
                std::printf("%.16f", sum / current_mod->prev.numel);
            } else {
                if ((current_mod->prev.numel == 1)){
                    if (current_mod->prev[0] == forward_order[i-1]){
                        if (current_mod->outside_input){
                            mem.add(recurr_init);
                        } // else do nothing
                    } else {
                        mem.reset(1);
                        mem.add(module_hist.vals[current_mod->prev[0]]->stored_out);
                    }                
                } else {
                    if (check_membership(current_mod->prev, forward_order[i-1])){
                        Matrix t = mem[0];
                        mem.reset(current_mod->prev.numel);
                        for (int j = 0; j < current_mod->prev.numel; j++){
                            if (current_mod->prev[j] == forward_order[i-1]){
                                mem.add(t);
                            } else {
                                mem.add(module_hist[current_mod->prev[j]]->stored_out);
                            }
                        }
                    } else {
                        mem.reset(current_mod->prev.numel);
                        for (int i = 0; i < current_mod->prev.numel; i++){
                            mem.add(module_hist[current_mod->prev[i]]->stored_out);
                        }
                    }
                }
                current_mod->forward(mem);
            }
        }
    }

    void backward(){
        // Goes through and updates the tree
        GradBlock * current_mod; 
        Matrix grad;
        Matrix grad_;
        float norm = 1/float(module_hist.vals[reverse_order[0]]->prev.numel);
        for (int i = 1; i < reverse_order.numel; i++){
            // std::cout<<i<<"_"<<reverse_order[i]<<" ";
            current_mod = module_hist.vals[reverse_order[i]];
            
            // if (i == 11){
            //     grad.print();
            // }
            if (current_mod->loss_node){
                current_mod->backward(grad, norm);
            } else {
                if (current_mod->use_imm_grad){
                    // if (i == 11){
                    //     current_mod->stored_grad.print();
                    // }
                    if (current_mod->store_grad){
                        grad += current_mod->stored_grad;
                    }
                    // if (i == 11){
                    //     grad.print();
                    // }
                } else {
                    grad = current_mod->stored_grad;
                }

                if (((!current_mod->mod_type[4]) & (!current_mod->mod_type[5])) | (current_mod->outside_input)){
                    // if (i == 11){
                    //     grad.print();
                    // }
                    current_mod->backward(grad);
                    // std::cout<<"0";
                } else {
                    // grad.print();
                    // if (i == 6){
                    //     grad.print();
                    //     grad_.print();
                    // }
                    current_mod->backward(grad, grad_);
                    // std::cout<<"_1";
                    // if (i == 6){
                    //     grad.print();
                    //     grad_.print();
                    // }
                }
            }

            if (current_mod->out_grad){
                if (current_mod->prev.numel == 1){
                    module_hist.vals[current_mod->prev[0]]->stored_grad = grad; 
                } else {
                    // std::cout<<std::endl<<"considering "<<reverse_order[i]<<": "<<grad.dim1<<" "<<grad.dim2<<" "<<grad_.dim1<<" "<<grad_.dim2<<std::endl;
                    // if (current_mod->mod_type[4]){
                    //     std::cout<<"res " << current_mod->prev[0] << " " <<current_mod->prev[1]<<std::endl;
                    // } else if (current_mod->mod_type[5]){
                    //     std::cout<<"concat " << current_mod->prev[0] << " " <<current_mod->prev[1]<<std::endl;
                    // }
                    //////////////////////////
                    if (current_mod->out_grad_[0]){
                        module_hist.vals[current_mod->prev[0]]->stored_grad = grad; 
                    }
                    if (current_mod->out_grad_[1]){
                        module_hist.vals[current_mod->prev[1]]->stored_grad = grad_;
                    }
                    if (current_mod->swap){
                        grad = grad_;
                    }
                    //////////////////////////

                    // if (i == 6){
                    //     grad.print();
                    //     grad_.print();
                    // }
                }
            }

            // if (current_mod->prev.numel == 1){
            //     grad.print();
            // } else {
            //     grad.print();
            //     grad_.print();
            // }

        }
    }
};


void clear_file(const char * tgt_file){
    std::ofstream write_file;
    write_file.open(tgt_file, std::ios::binary|std::ios::trunc); 
    write_file.close();
}

void write_matrix(const char * tgt_file, Matrix & to_write){
    std::ofstream write_file;
    write_file.open(tgt_file, std::ios::binary|std::ios::app); 
    char * temp = new char [30];
    float * buf_ = to_write.buf;

    write_file<<"[[";
    std::snprintf(temp, 30, "%.16f", *buf_);
    write_file << temp;
    buf_++;
    for (uint32_t j = 1; j < to_write.dim2; j++){
        std::snprintf(temp, 30, "%.16f", *buf_);
        write_file << ", "<< temp;
        buf_++;
    }
    write_file << "]";
    for (uint32_t i = 1; i < to_write.dim1; i++){
        std::snprintf(temp, 30, "%.16f", *buf_);
        write_file << ", [" << temp;
        buf_++;
        for (uint32_t j = 1; j < to_write.dim2; j++){
            std::snprintf(temp, 30, "%.16f", *buf_);
            write_file << ", "<< temp;
            buf_++;
        }
        write_file << "]";
    }
    write_file << "],";
    delete [] temp;
    write_file.close();
}

void write_model(array<LeftLinearB>& itm1, array<RightLinearB>& itm2, Embedding & embd, LeftLinearB & pc){
    const char * test_f = "test";
    clear_file(test_f);

    // itm1[itm1.numel-1].grad_weight.print();
    for (int i = 0; i < itm1.numel; i++){
        write_matrix(test_f, itm1[i].weight);
    }
    for (int i = 0; i < itm2.numel; i++){
        write_matrix(test_f, itm2[i].weight);
        write_matrix(test_f, itm2[i].bias);
    }
    write_matrix(test_f, embd.embd);
    write_matrix(test_f, embd.pos_emb);
    write_matrix(test_f, pc.weight);
}

void write_grad(array<LeftLinearB>& itm1, array<RightLinearB>& itm2, Embedding & embd, LeftLinearB & pc){
    const char * test_f = "grad";
    clear_file(test_f);

    // itm1[itm1.numel-1].grad_weight.print();
    for (int i = 0; i < itm1.numel; i++){
        write_matrix(test_f, itm1[i].grad_weight);
    }
    for (int i = 0; i < itm2.numel; i++){
        write_matrix(test_f, itm2[i].grad_weight);
        write_matrix(test_f, itm2[i].grad_bias);
    }
    write_matrix(test_f, embd.grad_embd);
    write_matrix(test_f, embd.grad_pos_emb);
    write_matrix(test_f, pc.grad_weight);
}


void step(array<LeftLinearB>& itm1, array<RightLinearB>& itm2, Embedding & embd, LeftLinearB & pc, bool mode){
    float lr = 0.01;

    // itm1[itm1.numel-1].grad_weight.print();
    for (int i = 0; i < itm1.numel; i++){
        itm1[i].update(lr, mode);
        itm1[i].zero_grad();
    }
    for (int i = 0; i < itm2.numel; i++){
        itm2[i].update(lr, mode);
        itm2[i].zero_grad();
    }
    embd.update(lr, mode);
    embd.zero_grad();
    pc.update(lr, mode);
    pc.zero_grad();

    write_model(itm1, itm2, embd, pc);
}


int main(){
    Tree t;
    std::cout<<t.module_hist.alloc_mem;
    int block_size = 2;
    int num_res = 5;
    int num_ctx = 5;

    uint32_t num_words = 100;
    uint32_t emb_size = 512;
    uint32_t sen_size = 32;

    int num_iter = 5;
    
    array<int> last_mod_idx(new int [10],10);
    array<int> cur_mod_idx(new int [10],10);
    int prev_mod_idx;
    int prev_recur_idx;

    ////////////////////////////////////////////
    // initialize model
    ////////////////////////////////////////////
    // Known bug: if size too short, will cause pointers to break when 
    // array is lengthened
    array<LeftLinearB> LeftList(new LeftLinearB [200], 200);
    array<RightLinearB> RightList(new RightLinearB [200], 200);

    selu selu_ = selu();
    residual res = residual();
    array<int> res_idx(new int [2], 2);
    concat concat_ = concat(0);
    array<int> concat_idx(new int [2], 2);
    CrossEntropy loss;
    array<int> loss_idx(new int [num_ctx], num_ctx);

    Embedding Embd(num_words,emb_size,sen_size);
    prev_mod_idx = t.add(&Embd, 0);

    prev_mod_idx = t.add(&concat_, prev_mod_idx, 0);
    
    LeftLinearB post_concat(2*sen_size, sen_size);
    prev_mod_idx = t.add(&post_concat, prev_mod_idx, false);

    for (int j = 0; j < num_res; j++){
        res_idx.reset(2);
        res_idx.add(prev_mod_idx);
        for (int i = 0; i < block_size; i++){
            prev_mod_idx = t.add(&selu_, prev_mod_idx, false);

            LeftList.add(LeftLinearB(sen_size, sen_size));
            prev_mod_idx = t.add(LeftList.vals+LeftList.numel-1, prev_mod_idx, false);

            RightList.add(RightLinearB(emb_size, emb_size, sen_size));
            prev_mod_idx = t.add(RightList.vals+RightList.numel-1, prev_mod_idx, false);
        }
        res_idx.add(prev_mod_idx);
        prev_mod_idx = t.add(&res, &res_idx, false);
    }
    concat_idx.set(prev_mod_idx, 1);

    prev_mod_idx = t.add(&selu_, prev_mod_idx, false);
    
    LeftList.add(LeftLinearB(sen_size, 1));
    prev_mod_idx = t.add(LeftList.vals+LeftList.numel-1, prev_mod_idx, false);
    
    RightList.add(RightLinearB(emb_size, num_words, 1));
    prev_mod_idx = t.add(RightList.vals+RightList.numel-1, prev_mod_idx, false);
    
    prev_mod_idx = t.add(&loss, prev_mod_idx, true);
    loss_idx.add(prev_mod_idx);

    for (int k = 0; k < num_ctx-1; k++){
        prev_mod_idx = t.add(&Embd, 1);

        concat_idx.set(prev_mod_idx, 0);
        prev_mod_idx = t.add(&concat_, &concat_idx, false);

        prev_mod_idx = t.add(&post_concat, prev_mod_idx, false);

        for (int j = 0; j < num_res; j++){
            res_idx.reset(2);
            res_idx.add(prev_mod_idx);
            for (int i = 0; i < block_size; i++){
                prev_mod_idx = t.add(&selu_, prev_mod_idx, false);

                prev_mod_idx = t.add(LeftList.vals+j*block_size+i, prev_mod_idx, false);

                prev_mod_idx = t.add(RightList.vals+j*block_size+i, prev_mod_idx, false);
            }
            res_idx.add(prev_mod_idx);
            prev_mod_idx = t.add(&res, &res_idx, false);
        }
        concat_idx.reset(2);
        concat_idx.set(prev_mod_idx, 1);

        prev_mod_idx = t.add(&selu_, prev_mod_idx, false);
    
        prev_mod_idx = t.add(LeftList.vals+num_res*block_size, prev_mod_idx, false);
        
        prev_mod_idx = t.add(RightList.vals+num_res*block_size, prev_mod_idx, false);
        
        prev_mod_idx = t.add(&loss, prev_mod_idx, true);
        loss_idx.add(prev_mod_idx);
    }
    Average av = Average();
    prev_mod_idx = t.add(&av, &loss_idx, false);

    t.compile();
    ////////////////////////////////////////////
    // Finish initialize model
    ////////////////////////////////////////////

    // // print model internal grad tree
    // std::cout<<std::endl<<std::endl<<"forwards (_: store_out, ^: outside_input)"<<std::endl;
    // for (int i = 0; i < t.forward_order.numel; i++){
    //     if (t.module_hist[t.forward_order[i]]->prev.numel != 1){
    //         std::cout<<"(";
    //         if (t.module_hist[t.forward_order[i]]->prev.numel != 0){
    //             for (int j = 0; j < t.module_hist[t.forward_order[i]]->prev.numel; j++){
    //                 std::cout<<t.module_hist[t.forward_order[i]]->prev[j]<<", ";
    //             }
    //         } else {
    //             std::cout<<"N";
    //         }
    //         std::cout<<")";
    //     } else if (t.module_hist[t.forward_order[i]]->prev[0] != t.forward_order[i-1]) {
    //         std::cout<<"("<<t.module_hist[t.forward_order[i]]->prev[0]<<")";
    //     }
    //     std::cout<<t.forward_order[i];
    //     if (t.module_hist[t.forward_order[i]]->next.numel != 1){
    //         std::cout<<"(";
    //         if (t.module_hist[t.forward_order[i]]->next.numel != 0){
    //             for (int j = 0; j < t.module_hist[t.forward_order[i]]->next.numel; j++){
    //                 std::cout<<t.module_hist[t.forward_order[i]]->next[j]<<", ";
    //             }
    //         } else {
    //             std::cout<<"N";
    //         }
    //         std::cout<<")";
    //     }
    //     if (t.module_hist[t.forward_order[i]]->store_out){
    //         std::cout<<"_";
    //     }
    //     if (t.module_hist[t.forward_order[i]]->outside_input){
    //         std::cout<<"^";
    //     }
    //     std::cout<<" ";
    // }
    // std::cout<<std::endl<<std::endl<<"backwards (_: not use_imm_grad, *: store_grad, ^: out_grad, *: out_grad)"<<std::endl;
    // for (int i = 0; i < t.reverse_order.numel; i++){
    //     if (!t.module_hist[t.reverse_order[i]]->use_imm_grad){
    //         std::cout<<"_";
    //     }
    //     if (t.module_hist[t.reverse_order[i]]->store_grad){
    //         std::cout<<"*";
    //     }
    //     if (t.module_hist[t.reverse_order[i]]->next.numel != 1){
    //         std::cout<<"(";
    //         if (t.module_hist[t.reverse_order[i]]->next.numel != 0){
    //             for (int j = 0; j < t.module_hist[t.reverse_order[i]]->next.numel; j++){
    //                 std::cout<<t.module_hist[t.reverse_order[i]]->next[j]<<", ";
    //             }
    //         } else {
    //             std::cout<<"N";
    //         }
    //         std::cout<<")";
    //     } else if (t.module_hist[t.reverse_order[i]]->next[0] != t.reverse_order[i-1]){
    //         std::cout<<"(" ;
    //         std::cout<<t.module_hist[t.reverse_order[i]]->next[0];
    //         std::cout<<")";
    //     }
    //     std::cout<<t.reverse_order[i];
    //     if (t.module_hist[t.reverse_order[i]]->prev.numel != 1){
    //         std::cout<<"(";
    //         if (t.module_hist[t.reverse_order[i]]->prev.numel != 0){
    //             for (int j = 0; j < t.module_hist[t.reverse_order[i]]->prev.numel; j++){
    //                 std::cout<<t.module_hist[t.reverse_order[i]]->prev[j];
    //                 if (t.module_hist[t.reverse_order[i]]->out_grad_[j]){
    //                     std::cout<<"^";
    //                 }
    //                 std::cout<<", ";
    //             }
    //         } else {
    //             std::cout<<"N";
    //         }
    //         std::cout<<")";
    //     } else if (t.module_hist[t.reverse_order[i]]->prev[0] != t.reverse_order[i+1]) {
    //         std::cout<<"("<<t.module_hist[t.reverse_order[i]]->prev[0];
    //         if (t.module_hist[t.reverse_order[i]]->out_grad_[0]){
    //             std::cout<<"^";
    //         }
    //         std::cout<<"), ";
    //     }
    //     if (t.module_hist[t.reverse_order[i]]->out_grad){
    //         std::cout<<"*";
    //     }
    //     std::cout<<" ";
    // }
    t.check_forward_backward();
    // finish print model internal grad tree

    //////////////////////////////////////////
    // Model forward
    //////////////////////////////////////////
    
    // Initialize test inputs
    array<IdxMatrix> inp(new IdxMatrix [num_ctx], num_ctx);

    uint32_t * d_t = new uint32_t [sen_size];
    int * tgt = new int [num_ctx];
    for (uint32_t i = 0; i < (uint32_t)num_ctx; i++){
        std::cout<<"printing d_t: ";
        tgt[i] = sen_size+i;
        for (uint32_t j = 0; j < sen_size; j++){
            d_t[j] = j + i;
            std::cout<<d_t[j]<<" ";
        }
        IdxMatrix d(sen_size, d_t);
        inp.add(d);
    }
    array<int> tgt_(tgt, num_ctx, num_ctx);
    std::cout<<" | printing tgt: ";
    for (int i = 0; i < tgt_.numel; i++){
        std::cout<<tgt_[i]<<" ";
    }
    std::cout<<std::endl;

    // write_model(LeftList, RightList, Embd, post_concat);

    for (int i = 0; i < num_iter; i++){
        Matrix m(sen_size, emb_size);
        t.forward(inp, tgt_, m);
        t.backward();
        write_grad(LeftList, RightList, Embd, post_concat);
        step(LeftList, RightList, Embd, post_concat, false);
    }
    return 0;
    
}
