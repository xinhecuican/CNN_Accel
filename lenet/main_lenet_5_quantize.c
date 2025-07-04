#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#define LABEL_LEN 2

void relu_int(int32_t *x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) {
            x[i] = 0;
        }
    }
}

void softmax(int32_t *x, float *output, int size) {
    int32_t max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    float sum = 0;
    for (int i = 0; i < size; i++) {
        output[i] = exp((float)(x[i] - max)) ;
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void conv_func(int8_t* d_i, int8_t* weight, int32_t* bias, int32_t* d_o, int8_t di_type,
                int height, int width, int padding, int channel, int out_channel, 
                int kernel_width, int kernel_height) {
	// 使用多重循环卷积(BUF_HEIGHT - KERNEL_SIZE + PADDING * 2) / STRIDE + 1
    int16_t* d_16 = (int16_t*)d_i;
    int32_t* d_32 = (int32_t*)d_i;
    int height_o = (height - kernel_height + padding * 2) + 1;
    int width_o = (width - kernel_width + padding * 2) + 1;
	for (int c = 0; c < out_channel; c++) {
		for (int h = -padding; h < (height - kernel_height + padding) + 1; h++) {
			for (int w = -padding; w < (width - kernel_width + padding) + 1; w++) {
                int32_t sum = 0;
                for(int k = 0; k < channel; k++) {
                    for (int m = 0; m < kernel_height; m++) {
                        for (int n = 0; n < kernel_width; n++) {
                            int h_m = h + m;
                            int w_n = w + n;
                            if (h_m >= 0 && h_m < height && w_n >= 0 && w_n < width) {
                                if(di_type == 0){
                                    sum += d_i[(k * height + h_m) * width + w_n] * 
                                        weight[((c * channel + k) * kernel_height + m) * kernel_width + n];
                                } else if (di_type == 1) {
                                    sum += d_16[(k * height + h_m) * width + w_n] * 
                                        weight[((c * channel + k) * kernel_height + m) * kernel_width + n];
                                } else if (di_type == 2) {
                                    sum += d_32[(k * height + h_m) * width + w_n] * 
                                        weight[((c * channel + k) * kernel_height + m) * kernel_width + n];
                                }
                            }
                        }
                    }
                }
                d_o[c * height_o * width_o + (h+padding) * width_o + w + padding] = sum + bias[c];
			}
		}
	}
}

void pool_func(int32_t* d_i, int32_t* d_o, int channel, int height, int width) {
	for (int c = 0; c < channel; c++) {
		for (int h = 0; h < height / 2; h++) {
			for (int w = 0; w < width / 2; w++) {
				int32_t sum = d_i[(c * height + (h << 1)) * width + (w << 1)] +
                             d_i[(c * height + (h << 1) + 1) * width + (w << 1)] +
                             d_i[(c * height + (h << 1)) * width + (w << 1) + 1] +
                             d_i[(c * height + (h << 1) + 1) * width + (w << 1) + 1];
				if(sum > 0) d_o[(c * height / 2 + h) * (width / 2) + w] = sum >> 2;
			}
		}
	}
}

void linear_func(
	int32_t* d_i, 
	int8_t* weight, 
	int32_t* bias, 
	int32_t* output, 
	int input_size, 
	int output_size) {
	
	for (int i = 0; i < output_size; i++) {
		output[i] = bias[i];
		for (int j = 0; j < input_size; j++) {
			output[i] += d_i[j] * weight[i * input_size + j];
		}
	}
}

void Prediction(int8_t image[28][28],
    int8_t w_conv1[6][3][3],
    int8_t w_conv2[16][6][3][3],
    int8_t w_fc1[10][576],
    int32_t b_conv1[6],
    int32_t b_conv2[16],
    int32_t b_fc1[10],
    float probs[10]) {


    // Conv1 layer
    int32_t conv1_out[6][28][28] = { 0 };
    conv_func(image, w_conv1, b_conv1, conv1_out, 0, 28, 28, 1, 1, 6, 3, 3);
    // for (int c = 0; c < 6; c++) {
    //     for (int h = -1, h2=0; h < 27; h++, h2++) {
    //         for (int w = -1, w2=0; w < 27; w++, w2++) {
    //             for (int m = 0; m < 3; m++) {
    //                 for (int n = 0; n < 3; n++) {
    //                     int h_m = h + m;
    //                     int w_n = w + n;
    //                     if (h_m < 0 || w_n < 0 || h_m >= 28 || w_n >= 28) continue;
    //                     conv1_out[c][h2][w2] += image[h_m][w_n] * w_conv1[c][m][n];
    //                 }
    //             }
    //             conv1_out[c][h2][w2] += b_conv1[c]; 
    //         }
    //     }
    // }


    // ReLU
    for (int c = 0; c < 6; c++) {
        relu_int(&conv1_out[c][0][0], 28 * 28);
    }

    // Pool1
    int32_t pool1_out[6][14][14] = { 0 };
    pool_func(conv1_out, pool1_out, 6, 28, 28);
    // for (int c = 0; c < 6; c++) {
    //     for (int h = 0; h < 14; h++) {
    //         for (int w = 0; w < 14; w++) {
    //             pool1_out[c][h][w] = (conv1_out[c][2*h][2*w] + conv1_out[c][2*h+1][2*w] + conv1_out[c][2*h][2*w+1] + conv1_out[c][2*h+1][2*w+1]);
    //           if(pool1_out[c][h][w] > 0) pool1_out[c][h][w] = pool1_out[c][h][w] >> 2; //right shift 2 bits ~ divide by 4
    //         }
    //     }
    // }

    // Conv2 layer
    int32_t conv2_out[16][12][12] = { 0 };
    conv_func(pool1_out, w_conv2, b_conv2, conv2_out, 2, 14, 14, 0, 6, 16, 3, 3);
    // for (int c = 0; c < 16; c++) {
    //     for (int h = 0; h < 12; h++) {
    //         for (int w = 0; w < 12; w++) {
    //             for (int k = 0; k < 6; k++) {
    //                 for (int m = 0; m < 3; m++) {
    //                     for (int n = 0; n < 3; n++) {
    //                         conv2_out[c][h][w] += pool1_out[k][h+m][w+n] * w_conv2[c][k][m][n];
    //                     }
    //                 }
    //             }
    //             conv2_out[c][h][w] += b_conv2[c];
    //         }
    //     }
    // }
        
    // ReLU
    for (int c = 0; c < 16; c++) {
        relu_int(&conv2_out[c][0][0], 12 * 12);
    }

    // Pool2
    int32_t pool2_out[16][6][6] = { 0 };
    pool_func(conv2_out, pool2_out, 16, 12, 12);
    // for (int c = 0; c < 16; c++) {
    //     for (int h = 0; h < 6; h++) {
    //         for (int w = 0; w < 6; w++) {
    //             pool2_out[c][h][w] = (conv2_out[c][2*h][2*w] + conv2_out[c][2*h+1][2*w] + conv2_out[c][2*h][2*w+1] + conv2_out[c][2*h+1][2*w+1]);
    //             if(pool2_out[c][h][w] > 0)  
    //                 pool2_out[c][h][w] = pool2_out[c][h][w] >> 2; //right shift 2 bits ~ divide by 4
    //         }
    //     }
    // }

    // Flatten
    int32_t flat_out[576] = { 0 };
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 6; h++) {
            for (int w = 0; w < 6; w++) {
                flat_out[c * 6 * 6 + h * 6 + w] = pool2_out[c][h][w];
            }
        }
    }

    // FC1
    int32_t fc1_out[10] = { 0 };
    linear_func(flat_out, w_fc1, b_fc1, fc1_out, 576, 10);
    // for (int i = 0; i < 10; i++) {
    //     for (int j = 0; j < 576; j++) {
    //         fc1_out[i] += w_fc1[i][j] * flat_out[j];
    //     }
    //     fc1_out[i] += b_fc1[i];
    // }

    // Softmax
    softmax(fc1_out, probs, 10);
}


int main(int argc, char** argv) {

    //float image[28][28];
    int8_t w_conv1[6][3][3];
    int8_t w_conv2[16][6][3][3];
    int8_t w_fc1[10][576];
    int32_t b_conv1[6];
    int32_t b_conv2[16];
    int32_t b_fc1[120];
    float probs[10];

    int i, j, m, n, index;
    FILE* fp;

    clock_t start, end;
    double cpu_time_used;
    /* Load Weights from DDR->LMM */
    fp = fopen("data/weights_int8/w_conv1.txt", "r");
    for (i = 0; i < 6; i++) {
        for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
                int temp;
                fscanf(fp, "%d ", &temp);
                w_conv1[i][m][n] = (int8_t)temp;
            }
        }
    }
    fclose(fp);

    fp = fopen("data/weights_int8/w_conv2.txt", "r");
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 6; j++) {
            for (m = 0; m < 3; m++) {
                for (n = 0; n < 3; n++) {
                    int temp;
                    fscanf(fp, "%d ", &temp);
                    w_conv2[i][j][m][n] = (int8_t)temp;
                }
            }
        }
    }
    fclose(fp);

    fp = fopen("data/weights_int8/w_fc1.txt", "r");
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 576; j++) {
            int temp;
            fscanf(fp, "%d ", &temp);
            w_fc1[i][j] = (int8_t)temp;
        }
    }
    fclose(fp);

    fp = fopen("data/weights_int8/b_conv1.txt", "r");
    for(i=0; i<6; i++) {
        float temp;
        fscanf(fp, "%f ", &temp);
        b_conv1[i] = (int32_t)temp;
    }    
    fclose(fp);
    
    fp = fopen("data/weights_int8/b_conv2.txt", "r");
    for(i=0; i<16; i++) {
        float temp;
        fscanf(fp, "%f ", &temp);
        b_conv2[i] = (int32_t)temp;
    }  
    fclose(fp);
    
    fp = fopen("data/weights_int8/b_fc1.txt", "r");
    for(i=0; i<120; i++) {
        float temp;
        fscanf(fp, "%f ", &temp);
        b_fc1[i] = (int32_t)temp;
    }
    fclose(fp);


    float* dataset = (float*)malloc(LABEL_LEN * 28 * 28 * sizeof(float));
    int target[LABEL_LEN];

    fp = fopen("mnist-test-target.txt", "r");
    for (i = 0; i < LABEL_LEN; i++)
        fscanf(fp, "%d ", &(target[i]));  fclose(fp);

    fp = fopen("mnist-test-image.txt", "r");
    for (i = 0; i < LABEL_LEN * 28 * 28; i++)
        fscanf(fp, "%f ", &(dataset[i]));  fclose(fp);

    float image[28][28];
    int8_t image_int[28][28];
    float* datain;
    int acc = 0;
    int mm, nn;

    start = clock();
    for (i = 0; i < LABEL_LEN; i++)
    {

        datain = &dataset[i * 28 * 28];
        for (mm = 0; mm < 28; mm++){
            for (nn = 0; nn < 28; nn++){
                image[mm][nn] = *(float*)&datain[28 * mm + nn];
                image_int[mm][nn] = image[mm][nn] * 3;
            }
        }

        Prediction(image_int,
            w_conv1,
            w_conv2,
            w_fc1,
            b_conv1,
            b_conv2,
            b_fc1,
            probs
        );

        int index = 0;
        float max = probs[0];
        for (j = 1; j < 10; j++) {
            if (probs[j] > max) {
                index = j;
                max = probs[j];
            }
        }

        if (index == target[i]) acc++;
        printf("Predicted label: %d\n", index);
        printf("Prediction: %d/%d\n", acc, i + 1);
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Accuracy = %f\n", acc*1.0f/LABEL_LEN);
    printf("Total inference time: %f seconds\n", cpu_time_used);
    printf("Average time per image: %f seconds\n", cpu_time_used/LABEL_LEN);
    free(dataset);

    return 0;
}

