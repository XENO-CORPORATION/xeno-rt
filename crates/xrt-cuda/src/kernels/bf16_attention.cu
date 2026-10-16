// Authored BF16 decode attention. MMA fragment layout follows NVIDIA PTX;
// the 128-key online softmax and normalization boundaries match the pinned
// PyTorch efficient-attention serving profile. No PyTorch code is linked.
__device__ float attention_bf16(float x) {
    unsigned int b=__float_as_uint(x);
    if((b&0x7f800000u)==0x7f800000u) return x;
    return __uint_as_float((b+0x7fff+((b>>16)&1))&0xffff0000u);
}
__device__ unsigned int attention_pair(float a,float b) {
    return (__float_as_uint(attention_bf16(a))>>16)|(__float_as_uint(attention_bf16(b))&0xffff0000u);
}
__device__ void attention_mma(float& a,float& b,float& c,float& d,unsigned int a0,unsigned int a1,unsigned int b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(a),"+f"(b),"+f"(c),"+f"(d) : "r"(a0),"r"(a1),"r"(b0));
}
extern "C" __global__ void native_attention_qk(const float* packed,float* scores,
    unsigned int heads,unsigned int kv,unsigned int dim,unsigned int rows) {
    unsigned int lane=threadIdx.x,g=lane/4,t=lane%4,group=heads/kv;
    unsigned int kvh=blockIdx.y,m=blockIdx.z*16+g,n=blockIdx.x*8+g;
    unsigned int qwidth=heads*dim;
    float c0=0,c1=0,c2=0,c3=0;
    for(unsigned int k=0;k<dim;k+=8) {
        unsigned int col=k+t*2;
        const float* q=packed+(kvh*group+m)*dim;
        const float* q8=q+8*dim;
        const float* key=packed+qwidth+(n*kv+kvh)*dim;
        unsigned int a0=m<group?attention_pair(q[col],q[col+1]):0;
        unsigned int a1=m+8<group?attention_pair(q8[col],q8[col+1]):0;
        unsigned int b0=n<rows?attention_pair(key[col],key[col+1]):0;
        attention_mma(c0,c1,c2,c3,a0,a1,b0);
    }
    unsigned int n0=blockIdx.x*8+t*2;
    float scale=dim==128?0.08838834764831845f:0.0625f;
    if(m<group) {
        unsigned int base=(kvh*group+m)*rows;
        if(n0<rows) scores[base+n0]=__fmul_rn(__fmul_rn(c0,scale),1.4426950408889634f);
        if(n0+1<rows) scores[base+n0+1]=__fmul_rn(__fmul_rn(c1,scale),1.4426950408889634f);
    }
    if(m+8<group) {
        unsigned int base=(kvh*group+m+8)*rows;
        if(n0<rows) scores[base+n0]=__fmul_rn(__fmul_rn(c2,scale),1.4426950408889634f);
        if(n0+1<rows) scores[base+n0+1]=__fmul_rn(__fmul_rn(c3,scale),1.4426950408889634f);
    }
}
extern "C" __global__ void native_attention_softmax(const float* scores,float* probabilities,float* stats,
    unsigned int rows,unsigned int blocks) {
    unsigned int head=blockIdx.x,lane=threadIdx.x,warp_n=lane/4,t=lane%4;
    __shared__ float maximum,denominator,rescale,parts[16];
    if(lane==0){maximum=-__int_as_float(0x7f800000);denominator=0;}
    __syncthreads();
    for(unsigned int block=0;block<blocks;++block) {
        unsigned int begin=block*128;
        if(lane==0) {
            float next=maximum;
            for(unsigned int i=begin;i<min(begin+128,rows);++i) next=fmaxf(next,scores[head*rows+i]);
            rescale=next>maximum?exp2f(maximum-next):1.0f;
            denominator=__fmul_rn(denominator,rescale);maximum=next;
        }
        __syncthreads();
        float sum=0;
        for(unsigned int tile=0;tile<4;++tile) for(unsigned int d=0;d<2;++d) {
            unsigned int i=begin+warp_n*32+t*2+tile*8+d;
            float value=i<rows?exp2f(__fsub_rn(scores[head*rows+i],maximum)):0.0f;
            if(i<rows) probabilities[head*rows+i]=attention_bf16(value);
            sum=__fadd_rn(sum,value);
        }
        parts[lane]=sum;__syncthreads();
        if(lane==0) {
            for(unsigned int group=0;group<4;++group) {
                unsigned int i=group*4;
                float total=__fadd_rn(__fadd_rn(parts[i],parts[i+1]),__fadd_rn(parts[i+2],parts[i+3]));
                denominator=__fadd_rn(denominator,total);
            }
            stats[(head*blocks+block)*2]=denominator;
            stats[(head*blocks+block)*2+1]=rescale;
        }
        __syncthreads();
    }
}
__device__ float attention_epilogue(float previous,float partial,float denominator,float rescale,
    bool first,bool last) {
    float alpha=last?1.0f/denominator:1.0f;
    if(first) return __fmul_rn(alpha,partial);
    float beta=__fmul_rn(alpha,rescale);
    return __fmaf_rn(alpha,partial,__fmul_rn(beta,previous));
}
extern "C" __global__ void native_attention_pv(const float* packed,const float* probability,
    const float* stats,float* out,unsigned int heads,unsigned int kv,unsigned int dim,unsigned int rows,unsigned int blocks) {
    unsigned int lane=threadIdx.x,g=lane/4,t=lane%4,group=heads/kv;
    unsigned int kvh=blockIdx.y,m=blockIdx.z*16+g,n=blockIdx.x*8+g;
    unsigned int h0=kvh*group+m,h8=h0+8;
    const float* values=packed+heads*dim+rows*kv*dim;
    float o0=0,o1=0,o2=0,o3=0;
    for(unsigned int block=0;block<blocks;++block) {
        float c0=0,c1=0,c2=0,c3=0;
        if(dim==128) {
            if(m<group) {float scale=stats[(h0*blocks+block)*2+1];c0=__fmul_rn(o0,scale);c1=__fmul_rn(o1,scale);}
            if(m+8<group) {float scale=stats[(h8*blocks+block)*2+1];c2=__fmul_rn(o2,scale);c3=__fmul_rn(o3,scale);}
        }
        for(unsigned int k=0;k<128;k+=8) {
            unsigned int key=block*128+k+t*2;
            float a=m<group&&key<rows?probability[h0*rows+key]:0;
            float b=m<group&&key+1<rows?probability[h0*rows+key+1]:0;
            float c=m+8<group&&key<rows?probability[h8*rows+key]:0;
            float d=m+8<group&&key+1<rows?probability[h8*rows+key+1]:0;
            float v0=key<rows&&n<dim?values[(key*kv+kvh)*dim+n]:0;
            float v1=key+1<rows&&n<dim?values[((key+1)*kv+kvh)*dim+n]:0;
            attention_mma(c0,c1,c2,c3,attention_pair(a,b),attention_pair(c,d),attention_pair(v0,v1));
        }
        if(dim==128) {
            o0=c0;o1=c1;o2=c2;o3=c3;
            if(block+1==blocks) {
                if(m<group){float inverse=1.0f/stats[(h0*blocks+block)*2];o0=__fmul_rn(o0,inverse);o1=__fmul_rn(o1,inverse);}
                if(m+8<group){float inverse=1.0f/stats[(h8*blocks+block)*2];o2=__fmul_rn(o2,inverse);o3=__fmul_rn(o3,inverse);}
            }
        } else {
            if(m<group){unsigned int i=(h0*blocks+block)*2;o0=attention_epilogue(o0,c0,stats[i],stats[i+1],block==0,block+1==blocks);o1=attention_epilogue(o1,c1,stats[i],stats[i+1],block==0,block+1==blocks);}
            if(m+8<group){unsigned int i=(h8*blocks+block)*2;o2=attention_epilogue(o2,c2,stats[i],stats[i+1],block==0,block+1==blocks);o3=attention_epilogue(o3,c3,stats[i],stats[i+1],block==0,block+1==blocks);}
        }
    }
    unsigned int col=blockIdx.x*8+t*2;
    if(m<group){if(col<dim)out[h0*dim+col]=attention_bf16(o0);if(col+1<dim)out[h0*dim+col+1]=attention_bf16(o1);}
    if(m+8<group){if(col<dim)out[h8*dim+col]=attention_bf16(o2);if(col+1<dim)out[h8*dim+col+1]=attention_bf16(o3);}
}
