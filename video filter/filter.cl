__kernel void filter(__global const float *image,
__global const float *filter,
__global float *restrict output,
const int rows,
const int cols,
const int filter_size)
{
int indexr=get_global_id(0);
int indexc=get_global_id(1);
int indexf=get_global_id(2);
float tmp=0.0;
for(int r=0;r<filter_size;r++){
for(int c=0;c<filter_size;c++){
tmp+=filter[r*filter_size+c]*image[indexf*(rows+filter_size-1)*(cols+filter_size-1)+indexr*(cols+filter_size-1)+indexc+r*(cols+filter_size-1)+c];}
}
output[indexf*rows*cols+indexr*cols+indexc]=tmp;
}


