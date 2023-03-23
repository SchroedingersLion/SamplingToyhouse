#ifndef PROBLEMS_H
#define PROBLEMS_H

#include <vector>
#define _USE_MATH_DEFINES 
#include <cmath>
#include <fstream>
#include <string>
#include <random>


class IPROBLEM {
/* 
The interface class of the problems to sample on (eg. double wells, harmonic oscillators, ...).
The particular problems are child classes inheriting from this class.
In particular, they need to implement the compute_force() routine.
*/

    public:

        std:: vector <double> parameters;            // Parameters, velocities, forces. 
        std:: vector <double> velocities;       
        std:: vector <double> forces;           
        
        // Constructor. Initializes the trajectories. 
        IPROBLEM(const std:: vector <double>& init_params, const std:: vector <double>& init_velocities)                         
            : parameters{init_params}, velocities{init_velocities}, forces{std:: vector <double> (parameters.size(),0)} {};   

        // Fills the force vector in each sampler iteration. Needs to be defined by the child classes.
        virtual void compute_force() = 0;           

        virtual ~IPROBLEM(){};                      // Destructor.

};




class HARMONIC_OSCILLATOR_1D: public IPROBLEM {
/*
This problem is the 1D harmonic oscillator with spring constant \omega^2.
*/

    private:

        const double omega_sq;      // Spring constant K = \omega^2.


    public:

        // Constructor.
        HARMONIC_OSCILLATOR_1D(const double omega_squared = 25, const std:: vector <double>& init_params = std:: vector <double> {0}, const std:: vector <double>& init_velocities = std:: vector <double> {0})
        : IPROBLEM(init_params, init_velocities), omega_sq{omega_squared} {
        }

        void compute_force() override {              
                                                               
            forces[0] = -omega_sq * parameters[0];
            
            return;

        };

};



class DOUBLE_GAUSSIAN_BASINS_2D: public IPROBLEM { 
/*
This problem is a 2D double well where the basins are of Gaussian shape. The minima lie at (-1,0) and (1,0).
For a visualization see the website.
*/    

    private:

        // Constants that define the potential.
        const std:: vector <double> mu1 {1, 0};		                // (x,y) of the first well. 
        const std:: vector <double> mu2 {-1, 0};		            // (x,y) of the second well.
        const std:: vector <double> SIG1 {0.6, 0.085, 0.02};	    // Elements of the two covariance matrices sig11, sig12, sig22 (note: sig12=sig21).
        const std:: vector <double> SIG2 {0.6, 0.085, 0.02};
        const double phi1 = 0.5;                                    // Mixing factor (phi2 = 1-phi1).	

        // Constants used by the force computation above.
        const double det1 = SIG1[0]*SIG1[2] - SIG1[1]*SIG1[1];      // Determinants of cov. matrices.
        const double det2 = SIG2[0]*SIG2[2] - SIG2[1]*SIG2[1];      
        const double inv_two_det_SIG1 = 1 / (2*det1) ;              // Used in the exponents in the force computation.
        const double inv_two_det_SIG2 = 1 / (2*det2) ;
        const double pref1 = phi1 / ( 2*M_PI*pow( det1, 1.5) );  	// Prefactors in the force computation.
        const double pref2 = (1-phi1) / ( 2*M_PI*pow( det2, 1.5) );


    public:

        // Constructor.
        DOUBLE_GAUSSIAN_BASINS_2D(const std:: vector <double>& init_params = std:: vector <double> {1,0}, const std:: vector <double>& init_velocities = std:: vector <double> {0,0})
        : IPROBLEM(init_params, init_velocities) {
        };


        void compute_force() override{      

            double x = parameters[0];       // Current positions.
            double y = parameters[1];

            double e1, e2, rho;             // Help constants.
            double x_mu_diff1 = x-mu1[0]; 
            double x_mu_diff2 = x-mu2[0]; 
            double y_mu_diff1 = y-mu1[1]; 
            double y_mu_diff2 = y-mu2[1];
            
            // The exponentials. 
            e1 = exp( -inv_two_det_SIG1 * ( SIG1[2]*x_mu_diff1*x_mu_diff1 - 2*SIG1[1]*x_mu_diff1*y_mu_diff1 + SIG1[0]*y_mu_diff1*y_mu_diff1 ) );
            e2 = exp( -inv_two_det_SIG2 * ( SIG2[2]*x_mu_diff2*x_mu_diff2 - 2*SIG2[1]*x_mu_diff2*y_mu_diff2 + SIG2[0]*y_mu_diff2*y_mu_diff2 ) );

            rho = pref1 * det1 * e1  +  pref2 * det2 * e2;  // the density.

            // Force part without noise, i.e. plain double well.
            forces[0] = -1/rho * ( pref1 * e1 * (SIG1[2]*x_mu_diff1 - SIG1[1]*y_mu_diff1)   +   pref2 * e2 * (SIG2[2]*x_mu_diff2 - SIG2[1]*y_mu_diff2) );
            forces[1] = -1/rho * ( pref1 * e1 * (-SIG1[1]*x_mu_diff1 + SIG1[0]*y_mu_diff1)  +   pref2 * e2 * (-SIG2[1]*x_mu_diff2 + SIG2[0]*y_mu_diff2) );

            // for noisy gradients --- TO BE IMPLEMENTED
            // if(sig!=0){                             
            //     normal_distribution<> normal{0,sig};	
            //     F.fx += normal(twister);
            //     F.fy += normal(twister);
            // }

            return;

        };

};



class BAYES_INFERENCE_MEANS_GAUSSMIX_2COMPONENTS_1D: public IPROBLEM { 
/*
This problem is the Bayesian inference of the two means of a 1D Gaussian mixture model with two components and given variances and mixing factors.
The constructor needs to read in the data set, a .csv file of a single column holding the data points. The prior is Gaussian with given variance (see members below).
*/

    private:

        // Members that need to be set by the constructor.
        const std:: vector <double> Xdata;
        const int batchsize; 
        std:: vector <int> idx_arr;
        std:: mt19937 twister;

        // Constants that define the potential.
        const double sig1 = 3;		            // Gaussian mixture params.
        const double sig2 = 0.5;		
        const double a1 = 0.8;
        const double a2 = 0.2;
        const double sig0 = 5;		            // Gaussian prior std.dev.

        // Constants used by the force computation.
	    const int size_minus_B = Xdata.size()-batchsize;
	    const double scale = Xdata.size() / double(batchsize);

        const double two_sigsig1 = 2*sig1*sig1;                     // Used in likelihood.
        const double two_sigsig2 = 2*sig2*sig2;
        const double pref_exp1 = a1/(sqrt(2*M_PI)*sig1);
        const double pref_exp2 = a2/(sqrt(2*M_PI)*sig2);
        const double F_scale_1 = a1/(sqrt(2*M_PI)*sig1*sig1*sig1);  // Used in force.
        const double F_scale_2 = a2/(sqrt(2*M_PI)*sig2*sig2*sig2);
        const double sigsig0 = sig0*sig0;	
        const size_t Xdata_size = Xdata.size();
        
        double e1, e2, likelihood, likeli_inv, x, x_minus_mu1, x_minus_mu2;  // Help constants.
        int help_int, idx;
        

        const std:: vector <double> read_dataset(std:: string filename){      // Read in the data set, used by constructor.
            
            std:: ifstream datasource(filename);
	        std:: vector <double> Xdata;
	        std:: string row;
	        while (getline(datasource, row)){
		        Xdata.push_back(stod(row));
	        }
            
            return Xdata;
        
        }


    public:

        // Constructor. 
        /* Not only does it need to read in the data points, but, for the potential use of data set subsampling during gradient computations,
           it also has to initialize the random number generator and a help array of indices needed for this purpose. */
        BAYES_INFERENCE_MEANS_GAUSSMIX_2COMPONENTS_1D(std:: string filename, const int batchsize, const int randomseed=0, 
                                                      const std:: vector <double>& init_params = std:: vector <double> {-4,3}, const std:: vector <double>& init_velocities = std:: vector <double>  {0,0}) 
        : IPROBLEM(init_params, init_velocities), Xdata{read_dataset(filename)}, batchsize{batchsize} {    
            
            idx_arr.resize(Xdata.size());           // List of indices, used for subsampling in stoch. grads.
            for (int i=0; i<Xdata.size(); ++i){		
		        idx_arr[i] = i;
	        }

            std:: seed_seq seq{1,20,3200,403,5*randomseed+1,12000,73667,9474+randomseed,19151-randomseed};  // Init. rng.
            std:: vector <std::uint32_t> seeds(1);
            seq.generate(seeds.begin(), seeds.end());
            twister.seed(seeds.at(0));
        
        }

        void compute_force() override {              // Fills force vector (potentially with stochastic gradients).             

            forces[0] = 0;
            forces[1] = 0;

            if(Xdata.size() != batchsize){	// In case of subsampling...
                
                /* idx_arr stores the possible indices of the vector Xdata.
                   this loop randomly chooses B of them and stores them
                   at the end of idx_arr. */
                for(size_t i = Xdata.size()-1;  i >= size_minus_B;  --i){ 

                    std:: uniform_int_distribution<> distrib(0, i);		// Recreates this in every iter... is there a better way?
                
                    idx = distrib(twister);
                    help_int = idx_arr[i];
                    idx_arr[i] = idx_arr[idx];
                    idx_arr[idx] = help_int; 

                }

            }

            for(int i = idx_arr.size()-1;  i >= size_minus_B;  --i){		/* Actual force evaluation. 
                                                                               The B data points to be considered are given 
                                                                               by the B last indices stored in idx_arr. */
            
                    x = Xdata[ idx_arr[i] ];
                    x_minus_mu1 = x-parameters[0];
                    x_minus_mu2 = x-parameters[1];

                    e1 = exp( -(x_minus_mu1)*(x_minus_mu1)/(two_sigsig1) );
                    e2 = exp( -(x_minus_mu2)*(x_minus_mu2)/(two_sigsig2) );
                    
                    likelihood = pref_exp1 * e1  +  pref_exp2 * e2;				// Likelihood of a single data point.
                    likeli_inv = 1/likelihood;

                    forces[0] += likeli_inv * e1 * (x_minus_mu1);
                    forces[1] += likeli_inv * e2 * (x_minus_mu2);
                        
            }


            forces[0] *= F_scale_1 * scale;
            forces[1] *= F_scale_2 * scale;
            
            forces[0] -= parameters[0]/(sigsig0);   // Prior part of the force.
            forces[1] -= parameters[1]/(sigsig0);

            return;

        };

};


#endif // PROBLEMS_H
