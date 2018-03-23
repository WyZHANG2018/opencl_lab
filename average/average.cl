__kernel void average(__global const float *x, 
                       
                        __global float *restrict z)
{
int index=get_global_id(0);
float tmp=0.0;
for(unsigned i=0;i<200;i++){
tmp+=x[index*200+i];
}
z[index]=tmp;
}
