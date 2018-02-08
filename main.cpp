#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <cmath>
#include <iostream>
#define PI 3.14159265

using namespace cv;
using namespace std;

Mat negative(Mat src){
	Mat output(src.rows,src.cols, CV_8UC1, Scalar(0));
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			output.at<uchar>(i,j)= 255- src.at<uchar>(i,j);
		}
	}
	return output;
}
Mat scale_nn(float hfactor,float wfactor, Mat src){
	float rows= ceil(hfactor*src.rows);
	float cols= ceil(wfactor*src.cols);
	Mat output(rows,cols, CV_8UC1, Scalar(0));
	float inew,jnew;
	for (int i=0;i<rows;i++){
		for (int j=0;j<cols;j++){
			inew= (i+0.5)/hfactor;
			jnew= (j+0.5)/wfactor;
			/*if (inew< floor(inew)+0.5)
				inew= floor(inew);
			else
				inew= ceil(inew);
			if (jnew< floor(jnew)+0.5)
				jnew= floor(jnew);
			else
				jnew= ceil(jnew);*/
			inew= floor(inew)+0.5;
			jnew= floor(jnew)+0.5;
			if (inew>=src.rows)
				inew= src.rows-0.5;
			if (jnew>=src.cols)
				jnew= src.cols-0.5;
			output.at<uchar>(i,j)= src.at<uchar>(inew-0.5,jnew-0.5);
		}
	}
	return output;
}
Mat translation(int x,int y, Mat src){
	Mat output(src.rows+x,src.cols+y, CV_8U, Scalar(0));
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			output.at<uchar>(i+x,j+y)= src.at<uchar>(i,j);		
		}
	}
	return output;
}
Mat shear_y(float a,Mat src){
	Mat output(src.rows+a*src.cols,src.cols, CV_8U, Scalar(0));
	for (int i=0;i<src.rows+a*src.cols;i++){
		for (int j=0;j<src.cols;j++){
			int temp= i-a*(src.cols-j);
			if (temp>=0 && temp<src.rows)
				output.at<uchar>(i,j)= src.at<uchar>(temp,j);		
		}
	}
	return output;
}
Mat shear_x(float a,Mat src){
	Mat output(src.rows,src.cols+a*src.rows, CV_8U, Scalar(0));
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols+a*src.rows;j++){
			int temp= j-a*(src.rows-i);
			if (temp>=0 && temp<src.cols)
				output.at<uchar>(i,j)= src.at<uchar>(i,temp);		
		}
	}
	return output;
}
Mat rotation(double theta, Mat src){
	float rows,cols;
	float r_rows = ceil(abs(src.rows*cos(theta)+src.cols*sin(theta)));
	float r_cols = ceil(abs(src.cols*cos(theta)+src.rows*sin(theta)));
	rows=r_rows;
	cols=r_cols;
	if (src.rows>rows)
		rows=src.rows;
	if (src.cols>cols)
		cols=src.cols;
	Mat output1(rows,cols, CV_8U, Scalar(0));
	//cout << rows << endl << cols << endl;
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			float inew= floor((rows-src.rows)/2 + i);
			float jnew= floor((cols-src.cols)/2 + j);
			output1.at<uchar>(inew,jnew)= src.at<uchar>(i,j);		
		}
	}
	Mat output(rows,cols, CV_8U, Scalar(0));
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols; j++){
			float inew = floor(cos(theta)*(i-rows/2) + sin(theta)*(j-cols/2));
			float jnew = floor(-sin(theta)*(i-rows/2) + cos(theta)*(j-cols/2));
			if (inew+rows/2>0 && inew+rows/2<rows && jnew+cols/2>0 && jnew+cols/2<cols){
				output.at<uchar>(i,j) = output1.at<uchar>(inew+rows/2,jnew+cols/2);
			}			
		}
	}
	/*Mat output(rows,cols, CV_8U, Scalar(0));
	for (int i=0;i<rows;i++){
		for (int j=0;j<cols;j++){
			float inew2= floor(abs((i-rows/2)*cos(theta)-(j-cols/2)*sin(theta)));
			float jnew2= floor(abs(i*sin(theta)+j*cos(theta)));
			if (inew2 < rows && jnew2 < cols && inew2>0 && jnew2>0)
				output.at<uchar>(inew2,jnew2)= src.at<uchar>(i+rows/2,j+cols/2);		
		}
	}*/
	return output;
}

Mat bitplane(Mat src,int plane){
	Mat output(src.rows,src.cols, CV_8UC1, Scalar(255));
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			int temp= (int) src.at<uchar>(i,j);
			int bit = temp>>(plane-1);
			int check= bit&&1;
			if (check==1){
				output.at<uchar>(i,j)=1<<(plane-1);
			}
		}
	}
	return output;

}

Mat scale_bilinear(float hfactor,float wfactor, Mat src){
	float rows= ceil(hfactor*src.rows);
	float cols= ceil(wfactor*src.cols);
	Mat output(rows,cols, CV_8U, Scalar(0));
	float inew,jnew;
	for (int i=0;i<rows;i++){
		for (int j=0;j<cols;j++){
			inew= (i+0.5)/hfactor;
			jnew= (j+0.5)/wfactor;
			float a1,a2,b1,b2,d1,d2;
			/*if (inew< 0.5)
				inew=0.5;
			if (inew>src.rows-0.5)
				inew=src.rows-0.6;*/
			if (inew>= floor(inew)+0.5){
				a1= floor(inew)+0.5;
				a2= a1+1;
				d1=inew-a1;
			}
			else{
				a1= floor(inew)-0.5;
				a2= a1+1;
				d1= inew-a1;
			}

			/*if (jnew< 0.5)
				jnew=0.5;
			if (jnew>src.cols-0.5)
				jnew=src.cols-0.6;*/

			if (jnew>= floor(jnew)+0.5){
				b1= floor(jnew)+0.5;
				b2= b1+1;
				d2=jnew-b1;
			}
			else{
				b1= floor(jnew)-0.5;
				b2= b1+1;
				d2= jnew-b1;
			}
			if (a1<0.5)
				a1=0.5;
			if (b1<0.5)
				b1=0.5;
			if (a2>=src.rows)
				a2=src.rows-0.5;
			if (b2>=src.cols)
				b2=src.cols-0.5;
			float result= (1-d1)*(1-d2)*src.at<uchar>(a1-0.5,b1-0.5)+ (1-d1)*d2*src.at<uchar>(a1-0.5,b2-0.5) + d1*(1-d2)*src.at<uchar>(a2-0.5,b1-0.5)+ d1*d2*src.at<uchar>(a2-0.5,b2-0.5);
			output.at<uchar>(i,j)= round(result);
		}
	}
	return output;
}

Mat histogram_ad(Mat src,int len){
	Mat output(src.rows,src.cols,CV_8UC1,Scalar(0));
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			int count[256]={0};
			for (int k=i-len;k<i+len;k++){
				for (int l=j-len;l<j+len;l++){
					if (k>=0 && k<src.rows && l>=0 && l<src.cols){
						int temp= (int) src.at<uchar>(k,l);
						count[temp]++;
					}
				}
			}
			int c_count[256];
			int sum=0;
			for (int k=0;k<256;k++){
				sum+=count[k];
				c_count[k]=sum;
				//cout << c_count[i] << endl;
			}
			int total=4*len*len;
			int temp= (int) src.at<uchar>(i,j);
			output.at<uchar>(i,j)= floor(255*c_count[temp]/total);
		}
	}

	return output;
}

Mat histogram_match(Mat src, Mat ref){
	int count[256]={0};
	Mat output(src.rows,src.cols,CV_8UC1,Scalar(0));
	for (int i=0;i<ref.rows;i++){
		for (int j=0;j<ref.cols;j++){
			int temp= (int) ref.at<uchar>(i,j);
			count[temp]++;
		}
	}
	int ref_arr[256];
	int sum=0;
	for (int i=0;i<256;i++){
		sum+=count[i];
		ref_arr[i]=sum;
		//cout << c_count[i] << endl;
	}
	int total=ref.rows*ref.cols;
	for (int i=0;i<256;i++){
		ref_arr[i]=floor(255*ref_arr[i]/total);
	}
	int count2[256]={0};
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			int temp= (int) src.at<uchar>(i,j);
			count2[temp]++;
		}
	}
	int src_arr[256];
	sum=0;
	int total2= src.rows*src.cols;
	for (int i=0;i<256;i++){
		sum+=count2[i];
		src_arr[i]=floor(255*sum/total);
	}
	int value=0;
	for (int i=0;i<256;i++){
		int min=256;
		value=0;
		for (int j=0;j<256;j++){
			if (src_arr[i]-ref_arr[j]==0){
				value=j;
				break;
			}
			else if(abs(src_arr[i]-ref_arr[j])<min){
				min=abs(src_arr[i]-ref_arr[j]);
				value=j;
			}
		}
		src_arr[i]=value;
	}
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			int temp= (int) src.at<uchar>(i,j);
			output.at<uchar>(i,j)=src_arr[temp];
		}
	}
	return output;
}
Mat histogram_eq(Mat src){
	int count[256]={0};
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			int temp= (int) src.at<uchar>(i,j);
			count[temp]++;
		}
	}
	int c_count[256];
	int sum=0;
	for (int i=0;i<256;i++){
		sum+=count[i];
		c_count[i]=sum;
		//cout << c_count[i] << endl;
	}
	int total=src.rows*src.cols;
	//cout << endl << endl << endl << total << endl;
	Mat output(src.rows,src.cols,CV_8UC1,Scalar(0));
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			int temp= (int) src.at<uchar>(i,j);
			output.at<uchar>(i,j)= floor(255*c_count[temp]/total);
		}
	}
	return output;
}
Mat logarithm(float c,Mat src){
	Mat output(src.rows,src.cols, CV_8UC1, Scalar(0));
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			float temp= c*log(src.at<uchar>(i,j)+1);
			if (temp>254)
				temp=255;
			if (temp<0)
				temp=0;
			output.at<uchar>(i,j)= temp;
		}
	}
	return output;
}

Mat gamma(float c,float gam,Mat src){
	Mat output(src.rows,src.cols, CV_8UC1, Scalar(0));
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			float temp= c*pow(src.at<uchar>(i,j),gam);
			if (temp>254)
				temp=255;
			if (temp<0)
				temp=0;
			output.at<uchar>(i,j)= temp;
		}
	}
	return output;
}
Mat plt(int initsrc, int initout,int finalsrc,int finalout, Mat src){
	Mat output(src.rows,src.cols, CV_8UC1, Scalar(255));
	int count=0;
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			int temp= (int) src.at<uchar>(i,j);
			//cout << temp << endl;
			if (temp<= initsrc){
				output.at<uchar>(i,j)=round(temp*initout/initsrc);
				//count++;
				//output.at<uchar>(i,j)=0;
			}
			else if (temp<finalsrc && temp>initsrc){
				float m=(finalout-initout)/(finalsrc-initsrc);
				output.at<uchar>(i,j)=round((temp-initsrc)*m + initout);
			}
			else if (temp>= finalsrc){
				float m=(255-finalout)/(255-finalsrc);
				output.at<uchar>(i,j)=round((temp-finalsrc)*m + initsrc);
				//output.at<uchar>(i,j)=255;
			}
			else{
				output.at<uchar>(i,j)=src.at<uchar>(i,j);
			}
			//cout << (int) output.at<uchar>(i,j) << endl;
		}
	}
	//cout << count;
	return output;
}

Mat tiepoint(Mat src){
	Mat output(4*src.rows,4*src.cols, CV_8UC1, Scalar(255));
	Mat matrix(8,8,CV_32F,Scalar(0));
	int x[4],y[4];
	cout << "Enter the 4 desired image coordinates(x,y)" << endl;
	for (int i=0;i<4;i++){
		cin >> x[i] >> y[i];
	}
	for (int i=0;i<4;i++){
		for (int j=0;j<2;j++){
			matrix.at<float>(2*i+j,0+4*j)=x[i];
			matrix.at<float>(2*i+j,1+4*j)=y[i];
			matrix.at<float>(2*i+j,2+4*j)=x[i]*y[i];
			matrix.at<float>(2*i+j,3+4*j)=1;
		}
	}
	Mat inverse=matrix.inv();
	Mat input(8,1,CV_32F,Scalar(0));
	cout << "Enter the 4 distorted image coordinates(x,y)" << endl;
	for (int i=0;i<8;i++){
		float temp;
		cin >> temp;
		input.at<float>(i,0)=temp;
	}
	//cout << input << endl;
	Mat c= inverse*input;
	//cout << c << endl;
	/*for (int i=0;i<2*src.rows;i++){
		for (int j=0;j<2*src.cols;j++){
			float inew=i*c.at<float>(0,0)+j*c.at<float>(1,0)+i*j*c.at<float>(2,0)+c.at<float>(3,0);
			float jnew=i*c.at<float>(4,0)+j*c.at<float>(5,0)+i*j*c.at<float>(6,0)+c.at<float>(7,0);
			if (inew>=0 && inew<src.rows && jnew>=0 && jnew<src.cols)
				output.at<uchar>(i,j)= src.at<uchar>(inew,jnew);
		}
	}*/
	float inew,jnew, itemp, jtemp;
	for (int i=0;i<4*src.rows;i++){
		for (int j=0;j<4*src.cols;j++){
			itemp= (i+0.5);
			jtemp= (j+0.5);
			inew= itemp*c.at<float>(0,0)+jtemp*c.at<float>(1,0)+itemp*jtemp*c.at<float>(2,0)+c.at<float>(3,0);
			jnew= itemp*c.at<float>(4,0)+jtemp*c.at<float>(5,0)+itemp*jtemp*c.at<float>(6,0)+c.at<float>(7,0);
			inew= floor(inew)+0.5;
			jnew= floor(jnew)+0.5;
			inew=inew-0.5;
			jnew=jnew-0.5;
			if (inew<src.rows && inew>=0 && jnew<src.cols && jnew>=0)
				output.at<uchar>(i,j)= src.at<uchar>(inew,jnew);
		}
	}
	int rowcount=0, colcount=0;
	for (int i=0;i<4*src.rows;i++){
		for (int j=0;j<4*src.cols;j++){
			int temp= (int) output.at<uchar>(i,j);
			if (temp!=255){
				rowcount++;
				break;
			}
		}
	}
	for (int j=0;j<4*src.cols;j++){
		for (int i=0;i<4*src.rows;i++){
			int temp= (int) output.at<uchar>(i,j);
			if (temp!=255){
				colcount++;
				break;
			}
		}
	}
	//cout << rowcount << endl << colcount << endl;
	Mat output1(rowcount,colcount, CV_8UC1, Scalar(0));
	for (int i=0;i<rowcount;i++){
		for (int j=0;j<colcount;j++){
			output1.at<uchar>(i,j)=output.at<uchar>(i,j);
		}
	}

	//cout << input << endl;
	return output1;
}
float rmse(Mat lib, Mat user){
	float total=0;
	float diff=0;
	for (int i=0;i<user.rows;i++){
		for (int j=0;j<user.cols;j++){
			if (i<lib.rows)
				diff= pow(user.at<uchar>(i,j)-lib.at<uchar>(i,j),2);
			else
				diff= pow(user.at<uchar>(i,j),2);
			//cout << diff << endl;
			total+=diff;
		}
	}
	//cout << total << endl;
	total= total/(user.rows*user.cols);
	float output= sqrt(total);
	return output;
}
Mat difference(Mat lib, Mat user){
	Mat output(user.rows,user.cols, CV_8U, Scalar(0));
	for (int i=0;i<user.rows;i++){
		for (int j=0;j<user.cols;j++){
			if (i<lib.rows)
				output.at<uchar>(i,j)= user.at<uchar>(i,j)-lib.at<uchar>(i,j);
			else
				output.at<uchar>(i,j)= user.at<uchar>(i,j);
		}
	}
	return output;
}



int main(int argc, char** argv ) {
	Mat image;
	string name;
	cout << "\nEnter the filename of the source image which has to be processed:" << endl;
	cin >> name;
	image = imread(name , 0);
	cout << "\n\n1 -> Image resizing\n2 -> Image rotation\n3 -> Image translation\n4 -> Image Shearing\n5 -> Image Negative\n6 -> Log Transformation\n7 -> Power-law/Gamma Transformation\n8 -> Piecewise Linear Transformation\n9 -> Bitplane Sclicing\n10 -> Image reconstruction using tie points\n11 -> Histogram equalization\n13 -> Adaptive Histogram equalization\n14 -> Histogram matching\n\nEnter one of the preceding keys to perform the corresponding task: ";
	int root;
	int next;
	Mat output(8,8,CV_8UC1,Scalar(0));
	Mat image2(8,8,CV_8UC1,Scalar(0));
	cin >> root;
	//namedWindow( "Input Image", WINDOW_AUTOSIZE );
	//namedWindow( "Output Image", WINDOW_AUTOSIZE );
	if (root==1){
		cout << "Enter the height and width factor by which the image has to be scaled: ";
		float hfactor, wfactor;
		cin >> hfactor >> wfactor;
		cout << "\nEnter 1 to use nearest neighbor interpolation or enter 2 to use bilinear interpolation: ";
		cin >> next;
		if (next==1){
			output=scale_nn(hfactor,wfactor,image);
			Mat output2;
			resize(image, output2, cv::Size(image.cols * wfactor,image.rows * hfactor), 0, 0, 0);
			float error= rmse(output2,output);
			cout << "\nOperation performed succesfully" << endl;
			cout << "The RMSE as compared to the inbuilt function for nearest neighbor scaling is " << error << endl;
		}
		else if (next==2){
			output=scale_bilinear(hfactor,wfactor,image);
			Mat output2;
			resize(image, output2, cv::Size(image.cols * wfactor,image.rows * hfactor), 0, 0, 1);
			float error= rmse(output2,output);
			cout << "\nOperation performed succesfully" << endl;
			cout << "The RMSE as compared to the inbuilt function for bilinear scaling is " << error << endl;
		}
	}
	else if (root==2){
		cout << "Enter the degree by which you want to rotate your image: ";
		float angle;
		cin >> angle;
		double theta= angle*PI/180;
		output= rotation(theta,image);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==3){
		cout << "Enter the x and y distance by which the image has to be translated: ";
		int x,y;
		cin >> x >> y;
		output= translation(x,y,image);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==4){
		cout << "Enter the value by which the image has to be sheared: ";
		float value;
		cin >> value;
		cout << "\nEnter 1 to shear along the x axis or enter 2 to shear along y axis: ";
		cin >> next;
		if (next==1){
			output= shear_x(value,image);
			cout << "\nOperation performed succesfully" << endl;
		}
		else if (next==2){
			output= shear_y(value,image);
			cout << "\nOperation performed succesfully" << endl;
		}
	}
	else if (root==5){
		output= negative(image);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==6){
		cout << "Enter value for log transformation: ";
		float c;
		cin >> c;
		output=logarithm(c,image);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==7){
		cout << "Enter c value and gamma value for gamma transformation: ";
		float c,gam;
		cin >> c >> gam;
		output=gamma(c,gam,image);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==8){
		cout << "Enter the two (x,y) mapping of intensity values (0-255) with x as source and y as the output intensity value: ";
		int initsrc, initout, finalsrc, finalout;
		cin >> initsrc >> initout >> finalsrc >> finalout;
		output=plt(initsrc,initout, finalsrc,finalout,image);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==9){
		cout << "Enter the bit value (1-8) which is to be displayed: ";
		int plane;
		cin >> plane;
		output= bitplane(image,plane);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==10){
		cout << "Enter the required 4 tie points: \n";
		output= tiepoint(image);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==11){
		output=histogram_eq(image);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==13){
		cout << "Enter the distance around which the adaptive histogram has to be calculated for single pixels: ";
		int len;
		cin >> len;
		output= histogram_ad(image,len);
		cout << "\nOperation performed succesfully" << endl;
	}
	else if (root==14){
		cout << "Enter the name of the reference image for the histogram matching: ";
		string name2;
		cin >> name2;
		image2=imread(name2,0);
		output= histogram_match(image,image2);
		cout << "\nOperation performed succesfully" << endl;
	}

	namedWindow( "Input Image", WINDOW_AUTOSIZE );
	namedWindow( "Output Image", WINDOW_AUTOSIZE );
	imwrite("output.jpeg",output);
	if (root==14){
		namedWindow( "Reference Image", WINDOW_AUTOSIZE );
		imshow("Reference Image",image2);
	}
	imshow("Input Image",image);
	imshow("Output Image",output);
	waitKey(0);
	/*int height= image.rows;
	int width= image.cols;
	Mat input(4,4, CV_8UC1, Scalar(0,0,255));
	int count=0;
	for (int i=0;i<4;i++){
		for (int j=0;j<4;j++){
			input.at<uchar>(i,j)= 100-count;
			count++;
		}
	}*/
	//image=input;
	//Mat output1= shear_x(0.5,image);
	//output1=image;
	//Mat output2;
	//float error= rmse(output2,output1);
	//Mat output3= difference(output2,output1);
	//cout << error << endl;
	//cout << input << endl << output1 << endl << output2 << endl;	
	//imshow( "OUTPUT", output1);
	//imshow( "INPUT", image );
	//imshow( "REFERENCE", image2 );
	//imshow( "Display window2", output2 );
	//imshow( "Display window2", output3 );
	return 0;
}



