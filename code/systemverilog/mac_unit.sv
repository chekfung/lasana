`timescale 1ns / 1ns
module mac_unit #(
    parameter N = 32,              // Number of inputs (32 for your case)
    parameter GAIN = 10,           // Differential amplifier gain
    parameter LOW_RESISTANCE = 78000, // Low resistance value (for high conductance)
    parameter HIGH_RESISTANCE = 202000 // High resistance value (for low conductance)
)(
    input real  in_voltage [N-1:0],   // Input voltages for the 32 inputs (real numbers)
    input real  weights [N-1:0],      // 32-bit weight values (-1, 0, +1) (real numbers)
    output real         out_voltage   // Output voltage (real number)
);

    // Internal signals
    real res_positive [N-1:0];  // Resistor values for positive memristors
    real res_negative [N-1:0];  // Resistor values for negative memristors
    real voltage_accumulated;   // Accumulated voltage for the dot product
    
    // Initialize resistors based on constant weights
    always @ (weights) begin
        for (int i = 0; i < N; i++) begin
            // If weight is 1, positive resistor has low resistance (high conductance), negative has high resistance
            //$display("Weight %d = %f", i, weights[i]);
            if (weights[i] == 1.0) begin
                res_positive[i] = LOW_RESISTANCE; // Low resistance (high conductance)
                res_negative[i] = HIGH_RESISTANCE; // High resistance (low conductance)
            end
            // If weight is -1, positive resistor has high resistance (low conductance), negative has low resistance
            else if (weights[i] == -1.0) begin
                res_positive[i] = HIGH_RESISTANCE; // High resistance (low conductance)
                res_negative[i] = LOW_RESISTANCE;  // Low resistance (high conductance)
            end
            // If weight is 0, both resistors have the same high resistance
            else begin
                res_positive[i] = HIGH_RESISTANCE; // High resistance (equal conductance)
                res_negative[i] = HIGH_RESISTANCE;
            end
        end
    end

    // Accumulate the output voltage by summing the weighted inputs
    always @(*) begin
        voltage_accumulated = 0.0;
        
        for (int i = 0; i < N; i++) begin
            // Use the input voltage directly, since it's already a real value
            //$display("Voltage: %f", in_voltage[i]);
            
            // Calculate voltage contribution for positive and negative resistors
            voltage_accumulated = voltage_accumulated + (in_voltage[i] * (1.0 / res_positive[i] - 1.0 / res_negative[i]));
        end
        //$display("Accumulated Voltage: %f", voltage_accumulated);
        out_voltage = voltage_accumulated * GAIN;
        //$display("Output Voltage: %f", out_voltage);
    end

endmodule
