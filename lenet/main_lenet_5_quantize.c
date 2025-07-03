#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include <math.h>
#include <stdint.h>
#include <time.h>

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

void Prediction(float image[28][28],
    int8_t w_conv1[6][3][3],
    int8_t w_conv2[16][6][3][3],
    int8_t w_fc1[10][576],
    int32_t b_conv1[6],
    int32_t b_conv2[16],
    int32_t b_fc1[10],
    float probs[10]) {


    // Conv1 layer
    int32_t conv1_out[6][28][28] = { 0 };
    for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 28; h++) {
            for (int w = 0; w < 28; w++) {
                for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                        conv1_out[c][h][w] += image[h][w] * w_conv1[c][m][n];
                    }
                }
                conv1_out[c][h][w] += b_conv1[c];
            }

        }

    }


    // ReLU
    for (int c = 0; c < 6; c++) {
        relu_int(&conv1_out[c][0][0], 28 * 28);
    }

    // Pool1
    int32_t pool1_out[6][14][14] = { 0 };
    for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 14; h++) {
            for (int w = 0; w < 14; w++) {
                pool1_out[c][h][w] = (conv1_out[c][2*h][2*w] + conv1_out[c][2*h+1][2*w] + conv1_out[c][2*h][2*w+1] + conv1_out[c][2*h+1][2*w+1]);
              if(pool1_out[c][h][w] > 0) pool1_out[c][h][w] = pool1_out[c][h][w] >> 2; //right shift 2 bits ~ divide by 4
            }
        }
    }

    // Conv2 layer
    int32_t conv2_out[16][12][12] = { 0 };
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 12; h++) {
            for (int w = 0; w < 12; w++) {
                for (int k = 0; k < 6; k++) {
                    for (int m = 0; m < 3; m++) {
                        for (int n = 0; n < 3; n++) {
                            conv2_out[c][h][w] += pool1_out[k][h+m][w+n] * w_conv2[c][k][m][n];
                        }
                    }
                }
                conv2_out[c][h][w] += b_conv2[c];
            }
        }
    }
        
    // ReLU
    for (int c = 0; c < 16; c++) {
        relu_int(&conv2_out[c][0][0], 12 * 12);
    }

    // Pool2
    int32_t pool2_out[16][6][6] = { 0 };
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 6; h++) {
            for (int w = 0; w < 6; w++) {
                pool2_out[c][h][w] = (conv2_out[c][2*h][2*w] + conv2_out[c][2*h+1][2*w] + conv2_out[c][2*h][2*w+1] + conv2_out[c][2*h+1][2*w+1]);
                if(pool2_out[c][h][w] > 0)  
                    pool2_out[c][h][w] = pool2_out[c][h][w] >> 2; //right shift 2 bits ~ divide by 4
            }
        }
    }

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
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 576; j++) {
            fc1_out[i] += w_fc1[i][j] * flat_out[j];
        }
        fc1_out[i] += b_fc1[i];
    }

    // Softmax
    softmax(fc3_out, probs, 10);
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
        int temp;
        fscanf(fp, "%d ", &temp);
        w_conv1[i][0][0] = (int8_t)temp;
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
    for (i = 0; i < 120; i++) {
        for (j = 0; j < 400; j++) {
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
    float* datain;
    int acc = 0;
    int mm, nn;

    start = clock();
    for (i = 0; i < LABEL_LEN; i++)
    {

        datain = &dataset[i * 28 * 28];
        for (mm = 0; mm < 28; mm++)
            for (nn = 0; nn < 28; nn++)
                image[mm][nn] = *(float*)&datain[28 * mm + nn];

        Prediction(image,
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

