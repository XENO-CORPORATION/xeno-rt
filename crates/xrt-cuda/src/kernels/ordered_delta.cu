// Eager FP32 gated-delta boundaries for a BF16 serving graph. State is
// key-major; head expansion is explicit. No text/CPU-reference kernel changes.
__device__ float delta_bf16(float x) {
    unsigned int b=__float_as_uint(x);
    if((b&0x7f800000u)==0x7f800000u) return x;
    return __uint_as_float((b+0x7fff+((b>>16)&1))&0xffff0000u);
}
extern "C" __global__ void ordered_delta_prepare(const float* in,float* normalized,
    unsigned int kh,unsigned int vh) {
    unsigned int h=blockIdx.x,lane=threadIdx.x,kwidth=kh*128,vwidth=vh*128;
    float qsum=0,ksum=0;
    #pragma unroll
    for(unsigned int d=0;d<4;++d) {
        unsigned int i=h*128+lane*4+d;
        qsum=__fadd_rn(qsum,__fmul_rn(in[i],in[i]));
        ksum=__fadd_rn(ksum,__fmul_rn(in[kwidth+i],in[kwidth+i]));
    }
    for(int offset=16;offset;offset/=2) {
        qsum=__fadd_rn(qsum,__shfl_down_sync(0xffffffff,qsum,offset));
        ksum=__fadd_rn(ksum,__shfl_down_sync(0xffffffff,ksum,offset));
    }
    float qi=rsqrtf(__shfl_sync(0xffffffff,qsum,0)+1e-6f);
    float ki=rsqrtf(__shfl_sync(0xffffffff,ksum,0)+1e-6f);
    #pragma unroll
    for(unsigned int d=0;d<4;++d) {
        unsigned int i=h*128+lane*4+d;
        normalized[i]=__fmul_rn(__fmul_rn(in[i],qi),0.08838834764831845f);
        normalized[kwidth+i]=__fmul_rn(in[kwidth+i],ki);
    }
    for(unsigned int value_head=h;value_head<vh;value_head+=kh) {
        if(lane==0) {
            unsigned int base=2*kwidth+vwidth;
            float a=in[base+value_head],b=in[base+vh+value_head];
            float bias=in[base+2*vh+value_head],log_a=in[base+3*vh+value_head];
            float t=__fadd_rn(a,bias);
            float softplus=t>20.0f?t:log1pf(expf(t));
            normalized[2*kwidth+value_head]=1.0f/(1.0f+expf(-b));
            normalized[2*kwidth+vh+value_head]=__fmul_rn(-expf(log_a),softplus);
        }
    }
}
extern "C" __global__ void ordered_delta_step(float* state,const float* in,
    const float* normalized,float* out,unsigned int kh,unsigned int vh,
    unsigned int tiled,unsigned int first) {
    unsigned int head=blockIdx.x,x=threadIdx.x,y=threadIdx.y;
    unsigned int source=tiled?head%kh:head/(vh/kh),kwidth=kh*128;
    float beta=normalized[2*kwidth+head],g=normalized[2*kwidth+vh+head];
    float decay=expf(g);
    float s[32][4],products[4][4];
    #pragma unroll
    for(unsigned int r=0;r<4;++r) for(unsigned int d=0;d<4;++d) products[r][d]=0;
    #pragma unroll
    for(unsigned int r=0;r<32;++r) {
        unsigned int key=y+r*4;
        float k=normalized[kwidth+source*128+key];
        #pragma unroll
        for(unsigned int d=0;d<4;++d) {
            unsigned int i=(head*128+key)*128+x*4+d;
            s[r][d]=__fmul_rn(state[i],decay);
            products[r%4][d]=__fadd_rn(products[r%4][d],__fmul_rn(s[r][d],k));
        }
    }
    __shared__ float reduction[4][128];
    __shared__ float delta[128];
    __shared__ float first_dot;
    #pragma unroll
    for(unsigned int d=0;d<4;++d)
        reduction[y][x*4+d]=__fadd_rn(__fadd_rn(__fadd_rn(products[0][d],products[1][d]),products[2][d]),products[3][d]);
    __syncthreads();
    for(unsigned int stride=2;stride;stride/=2) {
        if(y<stride) for(unsigned int d=0;d<4;++d)
            reduction[y][x*4+d]=__fadd_rn(reduction[y][x*4+d],reduction[y+stride][x*4+d]);
        __syncthreads();
    }
    if(y==0) {
        #pragma unroll
        for(unsigned int d=0;d<4;++d) {
            unsigned int vi=x*4+d;
            delta[vi]=__fmul_rn(__fsub_rn(in[2*kwidth+head*128+vi],reduction[0][vi]),beta);
        }
        if(first) {
            float dot=0;
            #pragma unroll
            for(unsigned int d=0;d<4;++d) {
                unsigned int key=x*4+d;
                dot=__fadd_rn(dot,__fmul_rn(normalized[source*128+key],normalized[kwidth+source*128+key]));
            }
            for(int offset=16;offset;offset/=2) dot=__fadd_rn(dot,__shfl_down_sync(0xffffffff,dot,offset));
            if(x==0) first_dot=dot;
        }
    }
    __syncthreads();
    #pragma unroll
    for(unsigned int r=0;r<4;++r) for(unsigned int d=0;d<4;++d) products[r][d]=0;
    #pragma unroll
    for(unsigned int r=0;r<32;++r) {
        unsigned int key=y+r*4;
        float k=normalized[kwidth+source*128+key],q=normalized[source*128+key];
        #pragma unroll
        for(unsigned int d=0;d<4;++d) {
            unsigned int vi=x*4+d,i=(head*128+key)*128+vi;
            float next=__fadd_rn(s[r][d],__fmul_rn(k,delta[vi]));
            state[i]=next;
            products[r%4][d]=__fadd_rn(products[r%4][d],__fmul_rn(next,q));
        }
    }
    #pragma unroll
    for(unsigned int d=0;d<4;++d)
        reduction[y][x*4+d]=__fadd_rn(__fadd_rn(__fadd_rn(products[0][d],products[1][d]),products[2][d]),products[3][d]);
    __syncthreads();
    for(unsigned int stride=2;stride;stride/=2) {
        if(y<stride) for(unsigned int d=0;d<4;++d)
            reduction[y][x*4+d]=__fadd_rn(reduction[y][x*4+d],reduction[y+stride][x*4+d]);
        __syncthreads();
    }
    if(y==0) {
        #pragma unroll
        for(unsigned int d=0;d<4;++d) {
            unsigned int vi=x*4+d;
            out[head*128+vi]=delta_bf16(first?__fmul_rn(first_dot,delta[vi]):reduction[0][vi]);
        }
        if(x==0) {
            out[vh*128+head]=beta;
            out[vh*128+vh+head]=g;
        }
    }
}
