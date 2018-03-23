__kernel void filter(__global const float *image,
__global const float *filter,
__global float *restrict output,
const int rows,
const int cols,
const int filter_size)
{
//int indexr=get_global_id(0);
//int indexc=get_global_id(1);
//int indexf=get_global_id(2);
int indexr_local=get_local_id(0);
int indexc_local=get_local_id(1);
int indexf_local=get_local_id(2);
int indexr_group=get_group_id(0);
int indexc_group=get_group_id(1);
int indexf_group=get_group_id(2);
float tmp=0.0;
for(int r=0;r<filter_size;r++){
for(int c=0;c<filter_size;c++){
tmp+=filter[r*filter_size+c]*image[indexf_group*(rows+filter_size-1)*(cols+filter_size-1)+(indexr_group*15+indexr_local)*(cols+filter_size-1)+(indexc_group*10+indexc_local)+r*(cols+filter_size-1)+c];}
}
output[indexf_group*rows*cols+(indexr_group*15+indexr_local)*cols+indexc_group*10+indexc_local]=tmp;
}


