module tb_leaky_integrate_and_fire #(parameter NUM_NEURONS=20000);

    parameter string FILE_PREFIX = "../data/spiking_20000_runs/pwl_files/spiking_20000_runs_pwl_file_";// Hyperparameter for the file prefix
    parameter bit ONLY_EVAL_ON_CLOCK = 0;
    parameter SIM_TIME = 500;

    initial begin 
        $display("NUM_NEURONS: %0d", NUM_NEURONS);
    end
    
    // Signals
    logic clk;
    logic rst;
    real i_in [0:NUM_NEURONS-1]; // Array of input currents for each LIF neuron
    logic spike [0:NUM_NEURONS-1];      // Spike signals for each LIF neuron

    // Instantiate SV-RNM modules
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : lif_instance
            leaky_integrate_and_fire lif (
                .clk(clk),
                .rst(rst),
                .i_in(i_in[i]),
                .spike(spike[i])
            );
        end
    endgenerate

    // Clock generation (200 MHz, #1 = .1 ns, so 5ns clock cycle)
    initial begin
        clk = 0;
        forever #25 clk = ~clk;
    end

    // Finish simulation after specified time units
    initial begin
        #SIM_TIME;
        $finish; // Finish simulation after SIM_TIME
    end

    // Simulation loop
    integer time_step = 0;
    integer max_time = 5000; // Maximum simulation time in time steps

    
    // Read and apply input waveform
    initial begin
        integer file_handle [0:NUM_NEURONS-1];
        integer neuron_id;
        real time_sec;
        real value;

        for (neuron_id = 0; neuron_id < NUM_NEURONS; neuron_id++) begin
            file_handle[neuron_id] = $fopen({FILE_PREFIX, $sformatf("%0d_I_inj.txt", neuron_id)}, "r");
        end

        while (time_step < max_time) begin
            // Update input currents for each neuron
            for (neuron_id = 0; neuron_id < NUM_NEURONS; neuron_id++) begin
                // Read time and value from file for neuron 'neuron_id'
                if ($feof(file_handle[neuron_id])) begin
                    $fclose(file_handle[neuron_id]);
                    break;
                end
                $fscanf(file_handle[neuron_id], "%f %f", time_sec, value);

                // COMMENT IF WE WANT TO ENABLE ONLY DURING CLOCK
                if (ONLY_EVAL_ON_CLOCK) begin
                    if (time_step % 40 == 0) begin
                        // Update input current for neuron 'neuron_id'
                        i_in[neuron_id] <= value;
                    end
                end else begin
                    // Update input current for neuron 'neuron_id'
                    i_in[neuron_id] <= value;
                end        
            end

            // Advance simulation time
            time_step++;
            #0.1; // Increment time step by 1 unit
        end

        // Close files
        for (neuron_id = 0; neuron_id < NUM_NEURONS; neuron_id++) begin
            $fclose(file_handle[neuron_id]);
        end
    end

endmodule