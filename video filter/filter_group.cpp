#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
using namespace cv;
using namespace std;
#define SHOW
#define STRING_BUFFER_LEN 1024

void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
		exit(1);
	}

unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
  *outputstr= (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("file size %d\n",size);
  printf("-------------------------------------------\n");
  snprintf((char *)*outputstr,size,"%s\n",*output);
  //printf("%s\n",*outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}

void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

// to get 2D gaussian kernel
cv::Mat getGaussianKernel2D(int rows, int cols, double sigmax, double sigmay )
{
        cv::Mat kernel = cv::Mat::zeros(rows, cols, CV_32FC1); 

        float meanj = (kernel.rows-1)/2, 
              meani = (kernel.cols-1)/2,
              sum = 0,
              temp= 0;

        float sigma=2*sigmay*sigmax;
        for(int  j=0;j<kernel.rows;j++)
            for(int  i=0;i<kernel.cols;i++)
            {
                temp = exp( -((j-meanj)*(j-meanj) + (i-meani)*(i-meani))  / (sigma));
                
                kernel.at<float>(j,i) = temp;

                sum += kernel.at<float>(j,i);
            }

        if(sum != 0)
            return kernel /= sum;
        else return cv::Mat();
}

//-------------------------------------------------------------------------
//--------------------------------------------------------------------------




int main(int, char**)
{
     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     { 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;




//--------------------------------------------------------------------
cl_mem image_buf; // num_devices elements
cl_mem filter_buf; // num_devices elements
cl_mem output_buf; // num_devices elements

int num_f=50; //number of frames
const int filter_size=31; //filter size 3*3, odd number
int status;


time_t start,end;
double diff;

time (&start);

    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME = "./output.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
   //Size S =Size(360,640);
   const int rows=(int) camera.get(CV_CAP_PROP_FRAME_HEIGHT);
   const int cols=(int) camera.get(CV_CAP_PROP_FRAME_WIDTH);

float *input_image=(float *) malloc(sizeof(float)*(rows+filter_size-1)*(cols+filter_size-1)*num_f);
float *input_filter=(float *) malloc(sizeof(float)*filter_size*filter_size);
float *output=(float *) malloc(sizeof(float)*rows*cols*num_f);



//-----------------------------------------------------------------------------------

    VideoWriter outputVideo;                                        // Open the output
    outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
	
	int count=0;
    //input image	
    while (true) {
        Mat cameraFrame;
		

        camera >> cameraFrame;
        Mat grayframe;
    	Mat grayframe_pad(Size(cols+filter_size-1,rows+filter_size-1),CV_8UC1);
        cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);
        copyMakeBorder(grayframe, grayframe_pad, (filter_size-1)/2,(filter_size-1)/2,(filter_size-1)/2,(filter_size-1)/2,BORDER_CONSTANT,0);
        std::copy(grayframe_pad.data,grayframe_pad.data+(rows+filter_size-1)*(cols+filter_size-1),input_image+(rows+filter_size-1)*(cols+filter_size-1)*count);

        count++;
        if(count >= num_f) break;

    }

    //input filter
    Mat mat_filter=getGaussianKernel2D(filter_size,filter_size, 1.0,1.0);
    //std::copy(mat_filter.data,mat_filter.data+filter_size*filter_size,input_filter);

for(int i=0;i<filter_size;i++){
for(int j=0;j<filter_size;j++){
input_filter[i*filter_size+j]=mat_filter.at<float>(i,j);
}
}

cout<<"filter matrix"<<endl;
for(int i=0;i<filter_size;i++){
for(int j=0;j<filter_size;j++){
cout<<input_filter[i*filter_size+j]<<" ";
}
cout<<endl;
}
cout<<endl;

    
time (&end);
diff = difftime (end,start);
printf ("CPU took %.2lf seconds to run.\n", diff );

//-----------------------------------------------------------------------------------

time (&start);

     clGetPlatformIDs(1, &platform, NULL);
     clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

     context_properties[1] = (cl_context_properties)platform;
     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
     context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
     queue = clCreateCommandQueue(context, device, 0, NULL);

     unsigned char **opencl_program=read_file("filter_group.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "filter", NULL);

//--------------------------------------------------------------------------


   // Input buffers.
    image_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       (rows+filter_size-1)*(cols+filter_size-1)*num_f* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    filter_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        filter_size*filter_size*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        rows*cols*num_f* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");


	cl_event write_event[2];
	cl_event kernel_event,finish_event;


    status = clEnqueueWriteBuffer(queue, image_buf, CL_FALSE,
        0, (rows+filter_size-1)*(cols+filter_size-1)*num_f* sizeof(float), input_image, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue,filter_buf, CL_FALSE,
        0, filter_size*filter_size*sizeof(float), input_filter, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &image_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &filter_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(kernel, argi++, sizeof(int), &rows);
checkError(status, "Failed to set argument 4");

    status = clSetKernelArg(kernel, argi++, sizeof(int), &cols);
checkError(status, "Failed to set argument 5");

    status = clSetKernelArg(kernel, argi++, sizeof(int), &filter_size);
checkError(status, "Failed to set argument 6");

//--------------------------------------------------------------------------------------
const size_t global_work_size[3] ={(size_t)rows,(size_t)cols,(size_t)num_f};
const size_t local_work_size[3] ={15,10,1};

    status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
        global_work_size, local_work_size, 2, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
0, rows*cols*num_f* sizeof(float), output, 1, &kernel_event, &finish_event);
   checkError(status, "Failed to Read Buffer");


time (&end);
diff = difftime (end,start);
printf ("GPU took %.2lf seconds to run.\n", diff );
//--------------------------------------------------------------------------------------
cout<<"number of frame per second "<<num_f/diff<<endl<<endl;
unsigned char* tmp=(unsigned char *) malloc(sizeof(unsigned char)*rows*cols);
for (int n=0;n<num_f;n++){

	for(int i=0;i<rows*cols;i++){
		tmp[i]=(unsigned char)((output+rows*cols*n)[i]);
	}

	Mat frame(rows,cols, CV_8UC1, tmp);
	Mat displayframe;
	cvtColor(frame, displayframe, CV_GRAY2BGR);
	//cout<<displayframe.cols<<" "<<displayframe.rows<<" "<<displayframe.type()<<" " << endl;
	outputVideo<< displayframe;
}

outputVideo.release();
camera.release();
// Release local events.
clReleaseEvent(write_event[0]);
clReleaseEvent(write_event[1]);
clReleaseKernel(kernel);
clReleaseCommandQueue(queue);
clReleaseMemObject(image_buf);
clReleaseMemObject(filter_buf);
clReleaseMemObject(output_buf);
clReleaseProgram(program);
clReleaseContext(context);

//--------------------------------------------------------------------

clFinish(queue);

return 0;


}
