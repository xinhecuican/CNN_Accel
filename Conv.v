`include "CNNConfig.vh"

module CNNConv (
    input clk,
    input rst,
    input stall,
    input [`WEIGHT_SIZE-1: 0][`WINDOW_SIZE-1: 0][31: 0] weight,

    input                               window_valid,
    input [`WINDOW_SIZE-1: 0][31: 0]    window,
    output                              window_stall,
    output [`WEIGHT_SIZE-1: 0][31: 0]   conv_data,
    output [`WEIGHT_SIZE-1: 0]          conv_valid
);

    reg [`WINDOW_SIZE+`WEIGHT_SIZE-1: 0] conv_valid_r;
    reg [`WINDOW_SIZE-1: 0][`WEIGHT_SIZE-1: 0][31: 0] res_r;

    wire stall_all      = (|conv_valid) & stall;
    assign window_stall = stall_all;

    always @(posedge clk)begin
        if(rst)begin
            conv_valid_r <= 0;
        end
        else if(!stall_all)begin
            conv_valid_r <= {window_valid, conv_valid_r[`WINDOW_SIZE+`WEIGHT_SIZE-1: 1]};
        end
    end

    assign conv_data = res_r[`WINDOW_SIZE-1];
    genvar i, j;
generate
    for(i=0; i<`WEIGHT_SIZE; i++)begin
        assign conv_valid[i] = conv_valid_r[`WEIGHT_SIZE-1-i];
    end
    for(i=0; i<`WINDOW_SIZE; i=i+1)begin
        reg [i: 0][31: 0] input_buffer;
        always @(posedge clk)begin
            if(~stall_all)begin
                if(window_valid)begin
                    input_buffer[i] <= window[i];
                end
                else begin
                    input_buffer[i] <= 0;
                end
            end
        end
        for(j=i-1; j>=0; j=j-1)begin
            always @(posedge clk)begin
                if(~stall_all) input_buffer[j] <= input_buffer[j+1];
            end
        end

        reg [`WEIGHT_SIZE-2: 0][31: 0] d_r;
        wire [`WEIGHT_SIZE-1: 0][31: 0] d_o;
        always @(posedge clk)begin
            if(~stall_all) d_r <= d_o[`WEIGHT_SIZE-2: 0];
        end
        for(j=0; j<`WEIGHT_SIZE; j=j+1)begin
            wire [31: 0] d_conv, d_i;
            if(i == 0)begin
                assign d_conv = 0;
            end
            else begin
                assign d_conv = res_r[i-1][j];
            end
            if(j == 0)begin
                assign d_i = input_buffer[0];
            end
            else begin
                assign d_i = d_r[j-1];
            end

            wire [31: 0] res;
            CNNPE #(i) pe (
                .d_conv(d_conv),
                .d(d_i),
                .w(weight[j][i]),
                .d_o(d_o[j]),
                .res(res)
            );
            always @(posedge clk)begin
                if(~stall_all) res_r[i][j] <= res;
            end
        end
    end
endgenerate

endmodule

module CNNPE #(
    parameter WITH_ADD=1
)(
    input [31: 0] d_conv,
    input [31: 0] d,
    input [31: 0] w,
    output [31: 0] d_o,
    output [31: 0] res
);
    wire signed [31: 0] mul_out;
    assign mul_out = $signed(d) * $signed(w);

generate
    if(WITH_ADD)begin
        assign res = $signed(d_conv) + $signed(mul_out);
    end
    else begin
        assign res = mul_out;
    end
endgenerate
    assign d_o = d;
endmodule