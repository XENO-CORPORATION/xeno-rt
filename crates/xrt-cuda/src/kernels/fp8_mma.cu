// SM89+ E4M3 Tensor Core projections. Fragment layout follows NVIDIA PTX ISA
// mma.m16n8k32: one live input row, eight output channels, other M rows zero.
// All tail loads are masked; original encoded weights remain byte-resident.
__device__ unsigned int fp8_byte(float value) {
    unsigned int bits=__float_as_uint(value),sign=(bits>>24)&128;
    if((bits&0x7f800000u)==0x7f800000u) return 127;
    float a=fabsf(value);
    if(a<0.015625f) return sign|(unsigned int)(a*512.0f);
    return sign|((((bits>>23)&255)-120)<<3)|((bits>>20)&7);
}
__device__ unsigned int pack_input(const float* x,unsigned int first,unsigned int cols) {
    unsigned int v=0;
    #pragma unroll
    for(unsigned int i=0;i<4;++i) if(first+i<cols) v|=fp8_byte(x[first+i])<<(8*i);
    return v;
}
__device__ unsigned int pack_weight(const unsigned char* w,unsigned int row,unsigned int first,unsigned int rows,unsigned int cols) {
    unsigned int v=0;
    #pragma unroll
    for(unsigned int i=0;i<4;++i) if(row<rows&&first+i<cols) v|=(unsigned int)w[(unsigned long long)row*cols+first+i]<<(8*i);
    return v;
}
__device__ void mma(float& c0,float& c1,float& c2,float& c3,unsigned int a0,unsigned int a2,unsigned int b0,unsigned int b1) {
    unsigned int zero=0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(zero),"r"(a2),"r"(zero),"r"(b0),"r"(b1));
}
__device__ float bf16(float value) {
    unsigned int b=__float_as_uint(value);
    if((b&0x7f800000u)==0x7f800000u) return value;
    return __uint_as_float((b+0x7fff+((b>>16)&1))&0xffff0000);
}
__device__ float silu(float value) {
    float exponential,exponent=value*(-1.4426950408889634f);
    asm("ex2.approx.f32 %0, %1;" : "=f"(exponential) : "f"(exponent));
    return value*(1.0f/(1.0f+exponential));
}
__device__ void group_dot(const float* x,const unsigned char* w,unsigned int row,unsigned int base,unsigned int rows,unsigned int cols,float& c0,float& c1) {
    unsigned int lane=threadIdx.x,group=lane>>2,t=lane&3;
    float c2=0.0f,c3=0.0f;c0=0.0f;c1=0.0f;
    #pragma unroll
    for(unsigned int k=0;k<128;k+=32) {
        unsigned int first=base+k+t*4;
        unsigned int a0=group==0?pack_input(x,first,cols):0;
        unsigned int a2=group==0?pack_input(x,first+16,cols):0;
        unsigned int b0=pack_weight(w,row+group,first,rows,cols);
        unsigned int b1=pack_weight(w,row+group,first+16,rows,cols);
        mma(c0,c1,c2,c3,a0,a2,b0,b1);
    }
}
extern "C" __global__ void fp8_mma_gate_up(const unsigned long long* p,const float* x,const float* activation_scale,
    float* hidden_out,unsigned int experts,unsigned int hidden,unsigned int middle) {
    unsigned int tile=blockIdx.x*8,expert=blockIdx.y,lane=threadIdx.x;
    const unsigned char* gate=(const unsigned char*)p[expert];
    const unsigned char* up=(const unsigned char*)p[experts+expert];
    const float* gs=(const float*)p[3*experts+expert];
    const float* us=(const float*)p[4*experts+expert];
    unsigned int groups=(hidden+127)/128;
    float g0=0,g1=0,u0=0,u1=0;
    for(unsigned int k=0;k<groups;++k) {
        float a,b,c,d;
        group_dot(x,gate,tile,k*128,middle,hidden,a,b);
        group_dot(x,up,tile,k*128,middle,hidden,c,d);
        if(lane<4) {
            unsigned int r=tile+lane*2;
            if(r<middle) {unsigned int s=(r/128)*groups+k;g0+=(a*activation_scale[k])*gs[s];u0+=(c*activation_scale[k])*us[s];}
            if(r+1<middle) {unsigned int s=((r+1)/128)*groups+k;g1+=(b*activation_scale[k])*gs[s];u1+=(d*activation_scale[k])*us[s];}
        }
    }
    if(lane<4) {
        unsigned int row=tile+lane*2;
        if(row<middle) {float g=bf16(g0),u=bf16(u0);hidden_out[(unsigned long long)expert*middle+row]=silu(g)*u;}
        if(row+1<middle) {float g=bf16(g1),u=bf16(u1);hidden_out[(unsigned long long)expert*middle+row+1]=silu(g)*u;}
    }
}
extern "C" __global__ void fp8_mma_down(const unsigned long long* p,const float* x,const float* activation_scale,
    const float* probability,float* output,unsigned int experts,unsigned int hidden,unsigned int middle,float denominator) {
    unsigned int tile=blockIdx.x*8,lane=threadIdx.x,groups=(middle+127)/128;
    unsigned int expert=blockIdx.y;
        const unsigned char* w=(const unsigned char*)p[2*experts+expert];
        const float* scales=(const float*)p[5*experts+expert];
        float total0=0,total1=0;
        for(unsigned int k=0;k<groups;++k) {
            float a,b;
            group_dot(x+(unsigned long long)expert*middle,w,tile,k*128,hidden,middle,a,b);
            if(lane<4) {
                unsigned int r=tile+lane*2;
                if(r<hidden) total0+=(a*activation_scale[expert*groups+k])*scales[(r/128)*groups+k];
                if(r+1<hidden) total1+=(b*activation_scale[expert*groups+k])*scales[((r+1)/128)*groups+k];
            }
        }
    if(lane<4) {
        unsigned int row=tile+lane*2;float route=bf16(probability[expert]/denominator);
        if(row<hidden) output[(unsigned long long)expert*hidden+row]=total0*route;
        if(row+1<hidden) output[(unsigned long long)expert*hidden+row+1]=total1*route;
    }
}
extern "C" __global__ void fp8_mma_reduce(const float* partial,float* output,unsigned int experts,unsigned int hidden) {
    unsigned int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=hidden) return;
    float sum=0;
    for(unsigned int expert=0;expert<experts;++expert) sum+=partial[(unsigned long long)expert*hidden+row];
    output[row]=sum;
}
