/* ---------------------------------------------------------------------
 *                                       _
 *  _ __ ___   __ _ _ __ _ __ ___   ___ | |_
 * | '_ ` _ \ / _` | '__| '_ ` _ \ / _ \| __|
 * | | | | | | (_| | |  | | | | | | (_) | |_
 * |_| |_| |_|\__,_|_|  |_| |_| |_|\___/ \__|
 *
 * Unit of Strength of Materials and Structural Analysis
 * University of Innsbruck
 * 2026 - today
 *
 * festigkeitslehre@uibk.ac.at
 *
 * This file is part of the MAteRialMOdellingToolbox (marmot).
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of marmot.
 * ---------------------------------------------------------------------
 */

#include "Marmot/FiniteStrainDruckerPrager.h"
#include "Marmot/MarmotDeformationMeasures.h"
#include "Marmot/MarmotEnergyDensityFunctions.h"
#include "Marmot/MarmotExceptions.h"
#include "Marmot/MarmotFastorTensorBasics.h"
#include "Marmot/MarmotMaterialFiniteStrain.h"
#include "Marmot/MarmotStressMeasures.h"

namespace Marmot::Materials {

  using namespace Marmot;
  using namespace Fastor;
  using namespace FastorIndices;
  using namespace FastorStandardTensors;

  FiniteStrainDruckerPrager::FiniteStrainDruckerPrager( const double* materialProperties,
                                                        int           nMaterialProperties,
                                                        int           materialLabel )
    : MarmotMaterialFiniteStrain( materialProperties, nMaterialProperties, materialLabel ),
      K( materialProperties[0] ),
      G( materialProperties[1] ),
      cohesion( materialProperties[2] ),
      phi( materialProperties[3] ),
      psi( materialProperties[4] ),
      A( materialProperties[5] ),
      hExpDelta( materialProperties[6] ),
      hExp( materialProperties[7] ),
      H( materialProperties[8] ),
      phiRad( Marmot::Math::degToRad( phi ) ),
      psiRad( Marmot::Math::degToRad( psi ) ),
      alpha_0( ( 6.0 * cohesion * std::cos( phiRad ) ) / ( std::sqrt( 3.0 ) * ( 3.0 + A * std::sin( phiRad ) ) ) ),
      beta( ( 6.0 * std::sin( phiRad ) ) / ( std::sqrt( 3.0 ) * ( 3.0 + A * std::sin( phiRad ) ) ) ),
      b( ( 6.0 * std::sin( psiRad ) ) / ( std::sqrt( 3.0 ) * ( 3.0 + A * std::sin( psiRad ) ) ) ),
      density( nMaterialProperties > 9 ? materialProperties[9] : 0.0 )
  {
    stateLayout.add( "Fp", 9 );     // plastic deformation gradient
    stateLayout.add( "alphaP", 1 ); // strain-like hardening variable
    stateLayout.finalize();
  }

  void FiniteStrainDruckerPrager::computeStress( ConstitutiveResponse< 3 >& response,
                                                 AlgorithmicModuli< 3 >&    tangents,
                                                 const Deformation< 3 >&    deformation,
                                                 const TimeIncrement&       timeIncrement ) const
  {
    TensorMap33d    Fp = stateLayout.getAs< TensorMap33d >( response.stateVars, "Fp" );
    const Tensor33d FpOld( Fp );
    double&         alphaP    = stateLayout.getAs< double& >( response.stateVars, "alphaP" );
    const double    alphaPOld = alphaP;

    using namespace Marmot;
    using namespace Fastor;
    using namespace Eigen;
    using namespace FastorIndices;
    using namespace FastorStandardTensors;

    Tensor33d FeTrial = deformation.F % Fastor::inverse( FpOld );
    double    betaP, dBetaP_dAlphaP;
    std::tie( betaP, dBetaP_dAlphaP ) = computeBetaP( alphaPOld );

    Tensor33d dFp;
    dFp.eye();
    Tensor33d Fe = FeTrial;

    if ( isYielding( FeTrial, betaP ) ) {
      int counter = 0;

      using mV9d = Eigen::Map< Eigen::Matrix< double, 9, 1 > >;
      VectorXd X( 11 );
      X.segment( 0, 9 )  = mV9d( FeTrial.data() );
      X( 9 )             = alphaPOld;
      X( 10 )            = 0;
      VectorXd        dX = VectorXd::Zero( 11 );
      VectorXd        R  = VectorXd::Zero( 11 );
      Eigen::MatrixXd dR_dX( 11, 11 );

      std::tie( R, dR_dX ) = computeResidualVectorAndTangent( X, FeTrial, alphaPOld );

      while ( R.norm() > 1e-12 || dX.norm() > 1e-12 ) {
        if ( counter > 10 )
          throw StressUpdateFailed( "inner newton not converged" );

        dX = -dR_dX.colPivHouseholderQr().solve( R );
        X += dX;
        std::tie( R, dR_dX ) = computeResidualVectorAndTangent( X, FeTrial, alphaPOld );
        counter += 1;
      }

      // update plastic deformation increment
      Fe              = X.segment( 0, 9 ).data();
      dFp             = Fastor::inverse( Fe ) % FeTrial;
      alphaP          = X( 9 );
      Tensor33d FpNew = dFp % Fp;
      memcpy( Fp.data(), FpNew.data(), 9 * sizeof( double ) );

      using namespace ContinuumMechanics;
      double      psi_;
      Tensor33d   Ce, dPsi_dCe;
      Tensor3333d dCe_dFe, d2Psi_dCedCe;
      std::tie( Ce, dCe_dFe ) = DeformationMeasures::FirstOrderDerived::rightCauchyGreen( Fe );

      // compute energy density, first and second partial derivatives wrt Cauchy Green deformation
      std::tie( psi_, dPsi_dCe, d2Psi_dCedCe ) = EnergyDensityFunctions::SecondOrderDerived::PenceGouPotentialB( Ce,
                                                                                                                 K,
                                                                                                                 G );

      // compute Kirchhoff stress
      Tensor33d   PK2 = 2. * dPsi_dCe;
      Tensor3333d dTau_dPK2, dTau_dFe_partial;
      std::tie( response.tau,
                dTau_dPK2,
                dTau_dFe_partial )  = StressMeasures::FirstOrderDerived::KirchhoffStressFromPK2( PK2, Fe );
      response.elasticEnergyDensity = psi_;
      auto [betaP_new, _]           = computeBetaP( alphaP );
      response.dissipation += 0.5 * ( betaP_new - betaP ) * ( alphaP - alphaPOld );

      // compute tangent operator
      using mM9d = Eigen::Map< Eigen::Matrix< double, 9, 9 > >;

      MatrixXd dYdDeformation              = MatrixXd::Zero( 11, 11 );
      dYdDeformation.block< 9, 9 >( 0, 0 ) = mM9d( Tensor3333d( einsum< IK, JL, to_IJKL >( Spatial3D::I,
                                                                                           transpose( Fastor::inverse(
                                                                                             FpOld ) ) ) )
                                                     .data() )
                                               .transpose();
      MatrixXd dXdDeformation = dR_dX.colPivHouseholderQr().solve( dYdDeformation );

      Tensor3333d dFe_dF = Tensor3333d( Matrix9d( dXdDeformation.block< 9, 9 >( 0, 0 ).transpose() ).data() );

      Tensor3333d dPK2_dFe = einsum< ijKL, KLMN >( 2. * d2Psi_dCedCe, dCe_dFe );
      Tensor3333d dPK2_dF  = einsum< ijKL, KLMN >( dPK2_dFe, dFe_dF );

      tangents.dTau_dF = einsum< IJKL, KLMN >( dTau_dPK2, dPK2_dF ) + einsum< ijKL, KLMN >( dTau_dFe_partial, dFe_dF );
    }
    else {
      using namespace Marmot::ContinuumMechanics;
      double      psi_;
      Tensor33d   Ce, dPsi_dCe;
      Tensor3333d dCe_dFe, d2Psi_dCedCe;
      std::tie( Ce, dCe_dFe ) = DeformationMeasures::FirstOrderDerived::rightCauchyGreen( Fe );

      std::tie( psi_, dPsi_dCe, d2Psi_dCedCe ) = EnergyDensityFunctions::SecondOrderDerived::PenceGouPotentialB( Ce,
                                                                                                                 K,
                                                                                                                 G );

      // compute Kirchhoff stress
      Tensor33d   PK2 = 2. * dPsi_dCe;
      Tensor3333d dTau_dPK2, dTau_dFe_partial;
      std::tie( response.tau,
                dTau_dPK2,
                dTau_dFe_partial )  = StressMeasures::FirstOrderDerived::KirchhoffStressFromPK2( PK2, Fe );
      response.elasticEnergyDensity = psi_;
      response.dissipation += 0.0;

      // compute tangent operator
      Tensor3333d dPK2_dFe = einsum< ijKL, KLMN >( 2. * d2Psi_dCedCe, dCe_dFe );
      Tensor3333d dFe_dF   = einsum< IK, JL, to_IJKL >( Spatial3D::I, transpose( Fastor::inverse( FpOld ) ) );
      Tensor3333d dPK2_dF  = einsum< ijKL, KLMN >( dPK2_dFe, dFe_dF );

      tangents.dTau_dF = einsum< IJKL, KLMN >( dTau_dPK2, dPK2_dF ) + einsum< ijKL, KLMN >( dTau_dFe_partial, dFe_dF );
    }
  }

  void FiniteStrainDruckerPrager::initializeYourself( double* stateVars, int nStateVars )
  {
    for ( int i = 0; i < nStateVars; ++i ) {
      stateVars[i] = 0.0;
    }
    TensorMap33d Fp = stateLayout.getAs< TensorMap33d >( stateVars, "Fp" );
    memcpy( Fp.data(), Spatial3D::I.data(), 9 * sizeof( double ) );
  }

} // namespace Marmot::Materials
