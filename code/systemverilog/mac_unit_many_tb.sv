`timescale 1ns / 1ns

module tb_mac_unit;

    // Parameters for the MAC unit
    parameter N = 32;             // Number of inputs
    parameter LOW_RESISTANCE = 78000;
    parameter HIGH_RESISTANCE = 202000;
    parameter GAIN = 10;
    parameter CLOCK_PERIOD = 4;   // Clock period in ns
    parameter NUM_MAC_UNITS = 20000; //  Number of MAC units to instantiate

    // Inputs for the MAC unit
    real in_voltage [NUM_MAC_UNITS-1:0][N-1:0];  // Array for input voltages
    real weights [NUM_MAC_UNITS-1:0][N-1:0];     // Array for weights
    
    // Outputs from the MAC units
    real out_voltage [NUM_MAC_UNITS-1:0];
    
    // Internal signals for clocking
    reg clk;                     // Clock signal
    reg reset;                   // Reset signal
    integer cycle_count;         // To count the number of clock cycles

    // Instantiate multiple MAC units using a generate block
    genvar i;
    generate
        for (i = 0; i < NUM_MAC_UNITS; i = i + 1) begin : mac_units
            mac_unit #(
                .N(N),
                .LOW_RESISTANCE(LOW_RESISTANCE),
                .HIGH_RESISTANCE(HIGH_RESISTANCE),
                .GAIN(GAIN)
            ) uut (
                .in_voltage(in_voltage[i]),
                .weights(weights[i]),
                .out_voltage(out_voltage[i])
            );
        end
    endgenerate

    // Clock generator
    always begin
        #(CLOCK_PERIOD / 2) clk = ~clk;  // Toggle clock every half period (2ns)
    end

    // Testbench procedure
    initial begin
        // Initialize clock and reset
        clk = 0;
        reset = 0;
        cycle_count = 0;

        // Set a random seed (optional)
        $random;
        
        // Initialize signals
        initialize_signals();
        
        // Run the test for a few cycles (up to 125 cycles = 500ns)
        for (cycle_count = 0; cycle_count < 125; cycle_count = cycle_count + 1) begin
            // Randomize the input voltages for all MAC units
            randomize_input_values();
            
            #CLOCK_PERIOD;
        end

        // End simulation after 500ns (125 clock cycles)
        $display("Simulation complete after 500ns.");
        $finish;
    end

    // Procedure to initialize the signals
    task initialize_signals;
        begin
            for (int j = 0; j < NUM_MAC_UNITS; j++) begin
                randomize_weights(j);
                for (int i = 0; i < N; i++) begin
                    in_voltage[j][i] = 0.0;
                end
            end
        end
    endtask

    task randomize_input_values;
        begin
            for (int j = 0; j < NUM_MAC_UNITS; j++) begin
                for (int i = 0; i < N; i++) begin
                    // Generate random value in the range [-0.8, 0.8]
                    in_voltage[j][i] = -0.8 + 1.6 * $urandom_range(0, 2147483647) / 2147483647.0;
                end
            end
        end
    endtask

    // Procedure to randomize the weights (-1, 0, +1)
    task randomize_weights(input integer unit_index);
        begin
            for (int i = 0; i < N; i++) begin
                // Randomize weights to be -1, 0, or 1
                case ($urandom % 3)
                    0: weights[unit_index][i] = -1.0;
                    1: weights[unit_index][i] = 0.0;
                    2: weights[unit_index][i] = 1.0;
                endcase
            end
        end
    endtask

endmodule
