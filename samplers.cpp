#include "samplers.h"


// ##### ISAMPLER METHODS ##### //

void ISAMPLER::run_mpi_simulation(int argc, char *argv[], const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const std:: string outputfile, const int t_meas){
    
    MPI_Init(&argc, &argv);				// Initialize MPI, use rank as random seed.
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nr_proc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nr_proc);

    const int seed = rank;

    print_sampler_params();
    draw_trajectory(max_iter, problem, RESULTS, seed, t_meas);    // Run sampler.

    std:: cout << "Rank " << rank << " reached barrier." << std:: endl;
    MPI_Barrier(comm);

    // Sum up results of different processors. 
    int row_size;

    for ( size_t i = 0;  i < RESULTS.measured_values.size();  ++i){
        
        row_size = RESULTS.measured_values[i].size();

        if( rank==0 ) RESULTS.measured_values_AVG[i].resize( row_size );     // Only on rank 0 to save RAM.

        MPI_Reduce( &RESULTS.measured_values[i][0], &RESULTS.measured_values_AVG[i][0], row_size, MPI_FLOAT, MPI_SUM, 0, comm);  // Collect results from processes.
        
        if( rank==0 ){
            for ( int j = 0;  j < row_size;  ++j ){
                RESULTS.measured_values_AVG[i][j] /= nr_proc;   // Divide by no. of processes to obtain averages.
            }
        }
    }	 

    if ( rank == 0) RESULTS.print_to_csv(t_meas, outputfile);     // Print to file, as specified by the measurement classes.
   
    MPI_Finalize();

    return;

};




void ISAMPLER::print_sampler_params(){    // Default behavior in case derived classes don't override.

    std::cout << "\n";

};




// ##### OBABO METHODS ##### //

void OBABO_sampler::print_sampler_params(){
    
    std:: cout << "OBABO sampling with parameters:\n";
    std:: cout << "Temperature = " << T << ",\nFriction = " << gamma << ",\nStepsize = " << h << ".\n" << std:: endl; 

};


void OBABO_sampler::draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const int randomseed, const int t_meas){

    // Integrator constants.
    const double a = exp(-1*gamma*h);    
    const double sqrt_a = sqrt(a);
    const double sqrt_aT = sqrt((1-a)*T);
    const double h_half = 0.5*h;   

    size_t No_params = problem.parameters.size();  // Number of parameters.

    // COMPUTE INITIAL FORCES.
    problem.compute_force();

    // COMPUTE INITIAL MEASUREMENTS.
    RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);

    // PREPARE RNG.
    std:: mt19937 twister;
    
    std:: seed_seq seq{1,20,3200,403,5*randomseed+1,12000,73667,9474+randomseed,19151-randomseed};
    std:: vector < std::uint32_t > seeds(1);
    seq.generate(seeds.begin(), seeds.end());
    twister.seed(seeds.at(0)); 

	std:: normal_distribution<> normal{0,1};
    double Rn;


	auto t1 = std:: chrono::high_resolution_clock::now();

    // MAIN LOOP.
    for ( size_t i = 1;  i <= max_iter;  ++i ) {

        // O + B steps.
        for ( size_t j = 0;  j < No_params;  ++j ) {                  
            
            Rn = normal(twister);
            problem.velocities[j] = sqrt_a * problem.velocities[j]  +  sqrt_aT * Rn  +  h_half * problem.forces[j]; 
        
        }

        // A step.
        for ( size_t j = 0;  j < No_params;  ++j ) {

            problem.parameters[j] += h * problem.velocities[j];
        
        }	
  
        // COMPUTE NEW FORCES.
        problem.compute_force();

        // B STEP.
        for ( size_t j = 0;  j < No_params;  ++j ) {

            problem.velocities[j] += h_half * problem.forces[j];
        
        }   
	
        // O STEP.
        for ( size_t j = 0;  j < No_params;  ++j ) {
            
		    Rn = normal(twister);
            problem.velocities[j] = sqrt_a * problem.velocities[j]  +  sqrt_aT * Rn;
        
        }   			

        // TAKE MEASUREMENT.
		if( i % t_meas == 0 ) {                                                      
            RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);    // Take measurement any t_meas steps.   
        }
		
        if( i % int(1e5) == 0 ) std:: cout << "Iteration " << i << " done!\n";
	
	}  // END MAIN LOOP.

    // FINALIZE.
    auto t2 = std:: chrono:: high_resolution_clock:: now();
	auto ms_int = std:: chrono:: duration_cast < std:: chrono:: seconds > (t2 - t1);
	std:: cout << "Execution took " << ms_int.count() << " seconds!\n";
        
    return;

};



// ##### BAOAB METHODS ##### //

void BAOAB_sampler::print_sampler_params(){
    
    std:: cout << "BAOAB sampling with parameters:\n";
    std:: cout << "Temperature = " << T << ",\nFriction = " << gamma << ",\nStepsize = " << h << ".\n" << std:: endl; 

};


void BAOAB_sampler::draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const int randomseed, const int t_meas){

    // Integrator constants.
    const double a = exp(-1*gamma*h);    
    const double sqrt_Ta_sq = sqrt((1-a*a)*T);
    const double h_half = 0.5*h;   

    size_t No_params = problem.parameters.size();  // Number of parameters.

    // COMPUTE INITIAL FORCES.
    problem.compute_force();

    // COMPUTE INITIAL MEASUREMENTS.
    RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);

    // PREPARE RNG.
    std:: mt19937 twister;
    
    std:: seed_seq seq{1,20,3200,403,5*randomseed+1,12000,73667,9474+randomseed,19151-randomseed};
    std:: vector < std::uint32_t > seeds(1);
    seq.generate(seeds.begin(), seeds.end());
    twister.seed(seeds.at(0)); 

	std:: normal_distribution<> normal{0,1};
    double Rn;


	auto t1 = std:: chrono::high_resolution_clock::now();

    // MAIN LOOP.
    for ( size_t i = 1;  i <= max_iter;  ++i ) {

            // B STEP.
            for ( size_t j = 0;  j < No_params;  ++j ) {

                problem.velocities[j] += h_half * problem.forces[j];
            
            }

            // A step.
            for ( size_t j = 0;  j < No_params;  ++j ) {

                problem.parameters[j] += h_half * problem.velocities[j];
            
            }

            // O STEP.
            for ( size_t j = 0;  j < No_params;  ++j ) {
                
                Rn = normal(twister);
                problem.velocities[j] = a * problem.velocities[j]  +  sqrt_Ta_sq * Rn;
            
            }   	

            // A step.
            for ( size_t j = 0;  j < No_params;  ++j ) {

                problem.parameters[j] += h_half * problem.velocities[j];
            
            }
        
            // COMPUTE NEW FORCES.
            problem.compute_force();

            // B STEP.
            for ( size_t j = 0;  j < No_params;  ++j ) {

                problem.velocities[j] += h_half * problem.forces[j];
            
            }

            // TAKE MEASUREMENT.
            if( i % t_meas == 0 ) {                                                      
                RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);    // Take measurement any t_meas steps.   
            }
            
            if( i % int(1e5) == 0 ) std:: cout << "Iteration " << i << " done!\n";
        
    }  // END MAIN LOOP.

    // FINALIZE.
    auto t2 = std:: chrono:: high_resolution_clock:: now();
	auto ms_int = std:: chrono:: duration_cast < std:: chrono:: seconds > (t2 - t1);
	std:: cout << "Execution took " << ms_int.count() << " seconds!\n";
        
    return;

};




// ##### SGHMC METHODS ##### //

void SGHMC_sampler::print_sampler_params(){
    
    std:: cout << "SGHMC sampling with parameters:\n";
    std:: cout << "Temperature = " << T << ",\nFriction = " << gamma << ",\nStepsize = " << h << ".\n" << std:: endl; 

};



void SGHMC_sampler::draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const int randomseed, const int t_meas){

    // Integrator constants.
    const double one_minus_hgamma = 1-h*gamma;    
    const double noise_pref = sqrt(2*h*gamma*T);  

    size_t No_params = problem.parameters.size();  // Number of parameters.

    // COMPUTE INITIAL FORCES.
    problem.compute_force();

    // COMPUTE INITIAL MEASUREMENTS.
    RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);

    // PREPARE RNG.
    std:: mt19937 twister;
    
    std:: seed_seq seq{1,20,3200,403,5*randomseed+1,12000,73667,9474+randomseed,19151-randomseed};
    std:: vector < std::uint32_t > seeds(1);
    seq.generate(seeds.begin(), seeds.end());
    twister.seed(seeds.at(0)); 

	std:: normal_distribution<> normal{0,1};
    double Rn;

	auto t1 = std:: chrono::high_resolution_clock::now();

    // MAIN LOOP.
    for ( size_t i = 1;  i <= max_iter;  ++i ) {
        
        // UPDATE.
        for ( size_t j = 0;  j < No_params;  ++j ) {                  
            
            Rn = normal(twister);
            problem.velocities[j] = one_minus_hgamma * problem.velocities[j]  +  noise_pref * Rn  +  h * problem.forces[j]; // Momentum update.
            problem.parameters[j] += h * problem.velocities[j];                                                             // Parameter update.
        
        }
  
        // COMPUTE NEW FORCES.
        problem.compute_force();
					 

        // TAKE MEASUREMENT.
		if( i % t_meas == 0 ) {                                                 
            RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);   // Take measurement any t_meas steps.        
		}
		
        if( i % int(1e5) == 0 ) std:: cout << "Iteration " << i << " done!\n";
	
	}  // END MAIN LOOP.


    // FINALIZE.
    auto t2 = std:: chrono:: high_resolution_clock:: now();
	auto ms_int = std:: chrono:: duration_cast < std:: chrono:: seconds > (t2 - t1);
	std:: cout << "Execution took " << ms_int.count() << " seconds!\n";
        
    return;

};





// ##### BBK_AMAGOLD METHODS ##### //

void BBK_AMAGOLD_sampler::print_sampler_params(){
    
    std:: cout << "BBK (AMAGOLD version) sampling with parameters:\n";
    std:: cout << "Temperature = " << T << ",\nFriction = " << gamma << ",\nStepsize = " << h << ".\n" << std:: endl; 

};



void BBK_AMAGOLD_sampler::draw_trajectory(const int max_iter, IPROBLEM& problem, IMEASUREMENT& RESULTS, const int randomseed, const int t_meas){

    std::cout<<"entering collect samples"<<std::endl;

    // Integrator constants.
    const double one_plus_hgamma_half_inv = 1 / (1+0.5*h*gamma);    
    const double one_minus_hgamma_half = 1-0.5*h*gamma;
    const double noise_pref = sqrt(2*h*gamma*T);  

    size_t No_params = problem.parameters.size();  // Number of parameters.

    // COMPUTE INITIAL FORCES.
    problem.compute_force();

    // COMPUTE INITIAL MEASUREMENTS.
    RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);

    // PREPARE RNG.
    std:: mt19937 twister;
    
    std:: seed_seq seq{1,20,3200,403,5*randomseed+1,12000,73667,9474+randomseed,19151-randomseed};
    std:: vector < std::uint32_t > seeds(1);
    seq.generate(seeds.begin(), seeds.end());
    twister.seed(seeds.at(0)); 

	std:: normal_distribution<> normal{0,1};
    double Rn;

	auto t1 = std:: chrono::high_resolution_clock::now();
    std::cout<<"starting main loop"<<std::endl;
   
    // MAIN LOOP.
    for ( size_t i = 1;  i <= max_iter;  ++i ) {
        
        // UPDATE.
        for ( size_t j = 0;  j < No_params;  ++j ) {                  
            
            problem.velocities[j] *= one_minus_hgamma_half;
            Rn = normal(twister);
            problem.velocities[j] = one_plus_hgamma_half_inv  *  ( problem.velocities[j]  +  h * problem.forces[j]  +  noise_pref * Rn );
            problem.parameters[j] += h * problem.velocities[j];                                                             
        
        }
  
        // COMPUTE NEW FORCES.
        problem.compute_force();
					 

        // TAKE MEASUREMENT.
		if( i % t_meas == 0 ) {                                                      
            RESULTS.take_measurement(problem.parameters, problem.velocities, problem.forces);   // Take measurement any t_meas steps.   
		}
		
        if( i % int(1e5) == 0 ) std:: cout << "Iteration " << i << " done!\n";
	
	}  // END MAIN LOOP.


    // FINALIZE.
    auto t2 = std:: chrono:: high_resolution_clock:: now();
	auto ms_int = std:: chrono:: duration_cast < std:: chrono:: seconds > (t2 - t1);
	std:: cout << "Execution took " << ms_int.count() << " seconds!\n";
        
    return;

};