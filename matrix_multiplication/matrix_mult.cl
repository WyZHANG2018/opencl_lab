__kernel void matrix_mult(__global const float *x,
                        __global const float *y,
                        __global float *restrict z){
int indexx=get_global_id(0);
int indexy=get_global_id(1);
float tmp=0.0;
for (unsigned i=0;i<1000;i++){
tmp+=x[indexx*1000+i]*y[i*1000+indexy];
}
z[indexx*1000+indexy]=tmp;
}
