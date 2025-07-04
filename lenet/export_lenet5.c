#include<stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#define LABEL_LEN 10000
#define EXPORT_SIZE 2

int main(int argc, char** argv) {
    int8_t w_conv1[6][3][3];
    int8_t w_conv2[16][6][3][3];
    int8_t w_fc1[10][576];
    int32_t b_conv1[6];
    int32_t b_conv2[16];
    int32_t b_fc1[10];

    int i, j, m, n, mm, nn;
    FILE* fp;

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
    for(i=0; i<10; i++) {
        float temp;
        fscanf(fp, "%f ", &temp);
        b_fc1[i] = (int32_t)temp;
    }
    fclose(fp);


    fp = fopen("data/weights.h", "w");
    fprintf(fp, "#ifndef __WEIGHTS__H__\n");
    fprintf(fp, "#define __WEIGHTS__H__\n\n");
    fprintf(fp, "#include <stdint.h>\n\n");
    fprintf(fp, "int8_t w_conv1[6][3][3] = {\n");
    for (i = 0; i < 6; i++) {
        fprintf(fp, "    {");
        for (m = 0; m < 3; m++) {
            fprintf(fp, "{");
            for (n = 0; n < 3; n++) {
                fprintf(fp, "%d", w_conv1[i][m][n]);
                if (n < 2) fprintf(fp, ", ");
            }
            fprintf(fp, "}");
            if (m < 2) fprintf(fp, ", ");
        }
        fprintf(fp, "}");
        if (i < 5) fprintf(fp, ",\n");
    }
    fprintf(fp, "\n};\n\n");
    fprintf(fp, "int8_t w_conv2[16][6][3][3] = {\n");
    for (i = 0; i < 16; i++) {
        fprintf(fp, "    {");
        for (j = 0; j < 6; j++) {
            fprintf(fp, "{");
            for (m = 0; m < 3; m++) {
                fprintf(fp, "{");
                for (n = 0; n < 3; n++) {
                    fprintf(fp, "%d", w_conv2[i][j][m][n]);
                    if (n < 2) fprintf(fp, ", ");
                }
                fprintf(fp, "}");
                if (m < 2) fprintf(fp, ", ");
            }
            fprintf(fp, "}");
            if (j < 5) fprintf(fp, ", ");
        }
        fprintf(fp, "}");
        if (i < 15) fprintf(fp, ",\n");
    }
    fprintf(fp, "\n};\n\n");
    fprintf(fp, "int8_t w_fc1[10][576] = {\n");
    for (i = 0; i < 10; i++) {
        fprintf(fp, "    {");
        for (j = 0; j < 576; j++) {
            fprintf(fp, "%d", w_fc1[i][j]);
            if (j < 575) fprintf(fp, ", ");
        }
        fprintf(fp, "}");
        if (i < 9) fprintf(fp, ",\n");
    }
    fprintf(fp, "\n};\n\n");
    fprintf(fp, "int32_t b_conv1[6] = {");
    for (i = 0; i < 6; i++) {
        fprintf(fp, "%d", b_conv1[i]);
        if (i < 5) fprintf(fp, ", ");
    }
    fprintf(fp, "};\n\n");
    fprintf(fp, "int32_t b_conv2[16] = {");
    for (i = 0; i < 16; i++) {
        fprintf(fp, "%d", b_conv2[i]);
        if (i < 15) fprintf(fp, ", ");
    }
    fprintf(fp, "};\n\n");
    fprintf(fp, "int32_t b_fc1[10] = {");
    for (i = 0; i < 10; i++) {
        fprintf(fp, "%d", b_fc1[i]);
        if (i < 9) fprintf(fp, ", ");
    }
    fprintf(fp, "};\n\n");
    fprintf(fp, "#endif // __WEIGHTS__H__\n");
    fclose(fp);

    printf("Weights exported to data/weights.h\n");

    // export test image(convert to int) and result to data/weight.h

    float* dataset = (float*)malloc(EXPORT_SIZE * 28 * 28 * sizeof(float));
    float* datain;
    float image[28][28];
    int8_t image_int[EXPORT_SIZE][28][28];
    int target[EXPORT_SIZE];
    fp = fopen("mnist-test-target.txt", "r");
    for (i = 0; i < EXPORT_SIZE; i++)
        fscanf(fp, "%d ", &(target[i]));  fclose(fp);

    fp = fopen("mnist-test-image.txt", "r");
    for (i = 0; i < EXPORT_SIZE * 28 * 28; i++)
        fscanf(fp, "%f ", &(dataset[i]));  fclose(fp);

    for (i = 0; i < EXPORT_SIZE; i++) {
        datain = &dataset[i * 28 * 28];
        for (mm = 0; mm < 28; mm++) {
            for (nn = 0; nn < 28; nn++) {
                image[mm][nn] = *(float*)&datain[28 * mm + nn];
                image_int[i][mm][nn] = image[mm][nn] * 3;
            }
        }
    }

    fp = fopen("data/test_image.h", "w");
    fprintf(fp, "#ifndef __TEST_IMAGE__H__\n");
    fprintf(fp, "#define __TEST_IMAGE__H__\n\n");
    fprintf(fp, "#include <stdint.h>\n\n");
    fprintf(fp, "#define TEST_IMAGE_SIZE %d\n\n", EXPORT_SIZE);
    fprintf(fp, "int8_t test_image[TEST_IMAGE_SIZE][28][28] = {\n");
    for (i = 0; i < EXPORT_SIZE; i++) {
        fprintf(fp, "    {");
        for (mm = 0; mm < 28; mm++) {
            fprintf(fp, "{");
            for (nn = 0; nn < 28; nn++) {
                fprintf(fp, "%d", image_int[i][mm][nn]);
                if (nn < 27) fprintf(fp, ", ");
            }
            fprintf(fp, "}");
            if (mm < 27) fprintf(fp, ", ");
        }
        fprintf(fp, "}");
        if (i < EXPORT_SIZE - 1) fprintf(fp, ",\n");
    }
    fprintf(fp, "\n};\n\n");
    fprintf(fp, "int test_target[TEST_IMAGE_SIZE] = {");
    for (i = 0; i < EXPORT_SIZE; i++) {
        fprintf(fp, "%d", target[i]);
        if (i < EXPORT_SIZE - 1) fprintf(fp, ", ");
    }   
    fprintf(fp, "};\n\n");
    fprintf(fp, "#endif // __TEST_IMAGE__H__\n");
    fclose(fp);

    printf("Test images exported to data/test_image.h\n");

    return 0;
}