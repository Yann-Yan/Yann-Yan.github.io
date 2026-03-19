=============================================================================
FYP Source Code: Robust Direct Learning Architecture for Digital Pre-Distortion
Author: Yann
Date: March 2026
=============================================================================

Dear Reader,

This directory contains the complete MATLAB source code for the Neural Network-based 
Digital Pre-Distortion (DPD) project. The codebase is fully modularized into 
Training and Verification phases to reflect standard software engineering practices.

-----------------------------------------------------------------------------
[ACTION REQUIRED]: FILE PATH CONFIGURATION
-----------------------------------------------------------------------------
Before executing any scripts, you MUST configure the local file paths. 
Please open the following scripts and replace the placeholder string 
'To be filled in' with the actual directory path on your local machine:

1. pa2.m        -> Set 'dataFolder' and 'saveFolder'
2. ILAtrain.m  -> Set 'dataFolder' and 'saveFolder'
3. ILAclose.m  -> Set 'dataFolder' and 'modelFolder'
4. DLAtrain.m  -> Set 'dataFolder' 
5. DLAclose.m  -> Set 'dataFolder' and 'modelFolder'

-----------------------------------------------------------------------------
EXECUTION WORKFLOW & DEPENDENCIES
-----------------------------------------------------------------------------
Please execute the scripts in the exact order below. 
The system relies on a strict save/load mechanism where the trained 
models (.mat files) are exported by the training scripts and subsequently 
imported by the verification scripts.

--- PHASE 1: The Physical Testbed ---
1. Run [ pa2.m ]
   - Function: Generates the baseband signal and models the highly nonlinear 
     Memory Polynomial Power Amplifier.
   - Output: Exports the dataset as 'nPA.mat'.
   *(Note: fpa2.m is the forward PA function called internally. Do not run it directly.)*

--- PHASE 2: Indirect Learning Architecture (ILA) ---
2. Run [ ILAtrain.m ]
   - Function: Trains the baseline BP-Neural Network using the ILA approach.
   - Output: Exports the trained architecture to 'ILA_model2.mat'.

3. Run [ ILAclose.m ]
   - Dependency: Requires 'nPA.mat' and 'ILA_model2.mat'.
   - Function: Performs closed-loop verification, demonstrating the initial 
     nonlinear correction and the physical saturation bottleneck.

--- PHASE 3: Direct Learning Architecture (DLA) - Final Proposed System ---
4. Run [ DLAtrain.m ]
   - Function: Executes the robust DLA iterative loop with dynamic annealing 
     and the anti-saturation Digital Shield (MAX_Z_AMP = 1.05).
   - Output: Exports the optimized architecture to 'DLA_model.mat'.

5. Run [ DLAclose.m ]
   - Dependency: Requires 'nPA.mat' and 'DLA_model.mat'.
   - Function: The ultimate verification script. Computes final Time-Domain (EVM) 
     and Frequency-Domain (ACLR) metrics, demonstrating Sub-1% EVM performance.

=============================================================================
End of README
=============================================================================