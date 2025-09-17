#include "Marmot/MarmotWiechertInterface.h"
#include <iostream>
#include <ostream>

namespace Marmot::Materials {

  using namespace Marmot;
  using namespace Eigen;

  namespace WiechertInterface {

    Properties generateRetardationTimes( int n, double min, double spacing )
    { //std::cout<<"Inside generateRetardationTimes"<<std::endl;
      Properties retardationTimes( n );
      for ( int i = 0; i < n; i++ )
        retardationTimes( i ) = min * std::pow( spacing, i );
      return retardationTimes;
    }

    void evaluateWiechert_Ru( double            dT,
                             Properties        elasticModuli_Ru,
                             Properties        retardationTimes_Ru,
                             StateVarMatrix_Ru stateVars_Ru,
                             double&           uniaxialStiffness_Ru,
                             Vector3d&         dforce_v,
                             const double      factor
                             )
    {
      for ( int i = 0; i < retardationTimes_Ru.size(); i++ ) {
        const double& tau = retardationTimes_Ru( i );
        const double& D   = elasticModuli_Ru( i );
        double lambda, beta;
        computeLambdaAndBeta( dT, tau, lambda, beta );
        uniaxialStiffness_Ru += lambda * D * factor;
        dforce_v += ( 1. - beta ) * stateVars_Ru.col( i ) * factor;
      }
    }
    
    void evaluateWiechert_Rs( double            dT,
                             Properties        elasticModuli_Rs,
                             Properties        retardationTimes_Rs,
                             StateVarMatrix_Rs stateVars_Rs,
                             double&           uniaxialStiffness_Rs,
                             Vector9d&         dsurface_stress_v,
                             const double      factor 
                            )
    {
      for ( int i = 0; i < retardationTimes_Rs.size(); i++ ) {
        const double& tau = retardationTimes_Rs( i );
        const double& D   = elasticModuli_Rs( i );

        double lambda, beta;
        computeLambdaAndBeta( dT, tau, lambda, beta );
        std::cout<<"lambda:\n"<<lambda<<"\n";
        std::cout<<"beta:\n"<<beta<<"\n";

        uniaxialStiffness_Rs += lambda * D * factor;
        dsurface_stress_v += ( 1. - beta ) * stateVars_Rs.col( i ) * factor;
      }
    }

    void updateStateVarMatrix_Ru( double                    dT,
                                  Properties                elasticModuli_Ru,
                                  Properties                retardationTimes_Ru,
                                  Ref< StateVarMatrix_Ru >  stateVars_Ru,
                                  const Vector3d&           dforce,
                                  const Matrix3d&           unitH_inv_ij
                                  )
    {

      if ( dT <= 1e-14 )
        return;
      for ( int i = 0; i < retardationTimes_Ru.size(); i++ ) {
        const double& tau = retardationTimes_Ru( i );
        const double& D   = elasticModuli_Ru( i );
        double        lambda, beta;
        computeLambdaAndBeta( dT, tau, lambda, beta );
        stateVars_Ru.col( i ) = ( lambda * D ) * unitH_inv_ij * dforce + beta * stateVars_Ru.col( i );
      }
    }

    void updateStateVarMatrix_Rs( double                   dT,
                                  Properties               elasticModuli_Rs,
                                  Properties               retardationTimes_Rs,
                                  Ref< StateVarMatrix_Rs > stateVars_Rs,
                                  const Vector9d&          dsurface_stress,
                                  const Matrix9d&          unitZ_ijkl )
    {

      if ( dT <= 1e-14 )
        return;
      for ( int i = 0; i < retardationTimes_Rs.size(); i++ ) {
        const double& tau = retardationTimes_Rs( i );
        const double& D   = elasticModuli_Rs( i );
        double        lambda, beta;
        computeLambdaAndBeta( dT, tau, lambda, beta );
        stateVars_Rs.col( i ) = ( lambda * D ) * unitZ_ijkl * dsurface_stress + beta * stateVars_Rs.col( i );
      }
    }

    void computeLambdaAndBeta( double dT, double tau, double& lambda, double& beta )
    {
      const double dT_tau = dT / tau;
      // respect extreme values according to Jirasek Bazant
      if ( dT_tau >= 30.0 ) {
        beta   = 0.;
        lambda = 1. / dT_tau;
      }
      else if ( dT_tau < 1e-6 ) {
        beta   = 1.0;
        lambda = 1 - 0.5 * dT_tau + 1. / 6 * dT_tau * dT_tau;
      }
      else {
        beta   = std::exp( -dT_tau );
        lambda = ( 1 - beta ) / dT_tau;
      }
    }

  } // namespace WienertInterface
} // namespace Marmot::Materials
