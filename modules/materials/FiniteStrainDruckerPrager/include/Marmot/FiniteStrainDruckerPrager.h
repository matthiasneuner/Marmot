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

#pragma once
#include "Marmot/MarmotDeformationMeasures.h"
#include "Marmot/MarmotEnergyDensityFunctions.h"
#include "Marmot/MarmotFastorTensorBasics.h"
#include "Marmot/MarmotFiniteStrainPlasticity.h"
#include "Marmot/MarmotJournal.h"
#include "Marmot/MarmotMaterialFiniteStrain.h"
#include "Marmot/MarmotMath.h"
#include "Marmot/MarmotTypedefs.h"
#include <stdexcept>
#include <string>
#include <tuple>

namespace Marmot::Materials {

  using namespace Fastor;
  using namespace FastorStandardTensors;
  using namespace FastorIndices;

  /**
   * @class Marmot::Materials::FiniteStrainDruckerPrager
   * @brief Finite-strain hyperelastic-plastic non-associated Drucker-Prager model.
   *
   * Elastic law: hyperelastic, compressible Neo-Hookean (Pence–Gou potential, variant B).
   */

  class FiniteStrainDruckerPrager : public MarmotMaterialFiniteStrain {
  public:
    using MarmotMaterialFiniteStrain::MarmotMaterialFiniteStrain;

    // elastic constants
    /** Bulk modulus (read from @c materialProperties[0]) */
    const double K;
    /** Shear modulus (read from @c materialProperties[1]) */
    const double G;

    // plasticity parameters
    /** Cohesion (read from @c materialProperties[2]) */
    const double cohesion;
    /** Friction angle (read from @c materialProperties[3]) */
    const double phi;
    /** Dilation angle (read from @c materialProperties[4]) */
    const double psi;
    /** Vertex scaling parameter (read from @c materialProperties[5]) */
    const double A;
    /** Hardening exponential delta (read from @c materialProperties[6]) */
    const double hExpDelta;
    /** Hardening exponential rate (read from @c materialProperties[7]) */
    const double hExp;
    /** Linear hardening modulus (read from @c materialProperties[8]) */
    const double H;

    // dependent parameters
    const double phiRad;
    const double psiRad;
    const double alpha_0;
    const double beta;
    const double b;

    // mass properties;
    /** Density (read from @c materialProperties[9]) (if provided). */
    const double density;

    /**
     * @brief Construct the finite-strain Drucker-Prager plasticity model.
     */
    FiniteStrainDruckerPrager( const double* materialProperties, int nMaterialProperties, int materialLabel );

    /**
     * @brief Compute the Kirchhoff stress and the algorithmic tangent.
     */
    void computeStress( ConstitutiveResponse< 3 >& response,
                        AlgorithmicModuli< 3 >&    tangents,
                        const Deformation< 3 >&    deformation,
                        const TimeIncrement&       timeIncrement ) const override;

    double getDensity( const double* stateVars ) const override
    {
      if ( this->nMaterialProperties < 10 ) {
        throw std::runtime_error(
          std::string( MakeString() << __PRETTY_FUNCTION__ << ": No density given! nMaterialProperties < 10." ) );
      }
      return this->density;
    }

    void initializeYourself( double* stateVars, int nStateVars ) override;

    /** @brief Yield function f(F^e, \beta_p) and first derivatives. */
    std::tuple< double, Tensor33d, double > yieldFunction( const Tensor33d& Fe, const double betaP ) const
    {
      Tensor33d   mandelStress;
      Tensor3333d dMandel_dFe;
      std::tie( mandelStress, dMandel_dFe ) = computeMandelStress( Fe );

      double      f;
      Tensor33d   df_dMandel;
      Tensor3333d d2f_dMandel_dMandel;
      Tensor33d   df_dFe;
      double      df_dBetaP;

      std::tie( f, df_dMandel, d2f_dMandel_dMandel, df_dBetaP ) = yieldFunctionFromStress( mandelStress, betaP );
      df_dFe                                                    = einsum< IJ, IJKL >( df_dMandel, dMandel_dFe );

      return { f, df_dFe, df_dBetaP };
    }

    /** @brief Yield function double version w.r.t. Mandel stress. */
    std::tuple< double, Tensor33d, Tensor3333d, double > yieldFunctionFromStress( const Tensor33d& mandelStress,
                                                                                  const double     betaP ) const
    {
      const double sigM = 1.0 / 3.0 * trace( mandelStress );
      Tensor33d    dev  = deviatoric( mandelStress );
      const double rho  = std::max( sqrt( Fastor::inner( dev, dev ) ), 1e-15 );
      const double q    = Constants::sqrt3_2 * rho;
      const double f    = q + Constants::sqrt3 * beta * sigM - betaP - Constants::sqrt3 * alpha_0;

      Tensor33d dRho_dMandel  = 1. / rho * dev;
      Tensor33d dSigM_dMandel = 1. / 3. * Spatial3D::I;
      Tensor33d df_dMandel    = Constants::sqrt3_2 * dRho_dMandel + Constants::sqrt3 * beta * dSigM_dMandel;

      Tensor3333d d2Rho_dMandel_dMandel = -1.0 / std::pow( rho, 3 ) * outer( dev, dev ) +
                                          1. / rho *
                                            ( einsum< ik, jl, to_ijkl >( Spatial3D::I, Spatial3D::I ) -
                                              1. / 3 *
                                                einsum< ij, IK, IL, to_ijKL >( Spatial3D::I,
                                                                               Spatial3D::I,
                                                                               Spatial3D::I ) );
      Tensor3333d d2f_dMandel_dMandel = Constants::sqrt3_2 * d2Rho_dMandel_dMandel;

      return { f, df_dMandel, d2f_dMandel_dMandel, -1.0 };
    }

    /** @brief Plastic potential gradient and Hessian (double version). */
    std::tuple< Tensor33d, Tensor3333d > plasticPotentialGradientAndHessianFromStress(
      const Tensor33d& mandelStress ) const
    {
      Tensor33d    dev = deviatoric( mandelStress );
      const double rho = std::max( sqrt( Fastor::inner( dev, dev ) ), 1e-15 );

      Tensor33d dRho_dMandel  = 1. / rho * dev;
      Tensor33d dSigM_dMandel = 1. / 3. * Spatial3D::I;
      Tensor33d dg_dMandel    = Constants::sqrt3_2 * dRho_dMandel + Constants::sqrt3 * b * dSigM_dMandel;

      Tensor3333d d2Rho_dMandel_dMandel = -1.0 / std::pow( rho, 3 ) * outer( dev, dev ) +
                                          1. / rho *
                                            ( einsum< ik, jl, to_ijkl >( Spatial3D::I, Spatial3D::I ) -
                                              1. / 3 *
                                                einsum< ij, IK, IL, to_ijKL >( Spatial3D::I,
                                                                               Spatial3D::I,
                                                                               Spatial3D::I ) );
      Tensor3333d d2g_dMandel_dMandel = Constants::sqrt3_2 * d2Rho_dMandel_dMandel;

      return { dg_dMandel, d2g_dMandel_dMandel };
    }

    bool isYielding( const Tensor33d& Fe, const double betaP ) const
    {
      double    f, df_dBetaP;
      Tensor33d df_dFe;
      std::tie( f, df_dFe, df_dBetaP ) = yieldFunction( Fe, betaP );
      return ( f > 0.0 );
    }

    std::tuple< Tensor33d, Tensor3333d > computeMandelStress( const Tensor33d& Fe ) const
    {
      using namespace Marmot::ContinuumMechanics;
      Tensor33d   Ce;
      Tensor3333d dCe_dFe;
      std::tie( Ce, dCe_dFe ) = DeformationMeasures::FirstOrderDerived::rightCauchyGreen( Fe );

      double      psi_;
      Tensor33d   dPsi_dCe;
      Tensor3333d d2Psi_dCedCe, dMandel_dCe;

      std::tie( psi_, dPsi_dCe, d2Psi_dCedCe ) = EnergyDensityFunctions::SecondOrderDerived::PenceGouPotentialB( Ce,
                                                                                                                 K,
                                                                                                                 G );
      Tensor33d       PK2                      = 2.0 * dPsi_dCe;
      const Tensor33d mandel                   = Ce % PK2;
      dMandel_dCe                              = einsum< Ii, iJKL, to_IJKL >( Ce, 2. * d2Psi_dCedCe ) +
                    einsum< IK, iL, iJ, to_IJKL >( Spatial3D::I, Spatial3D::I, PK2 );
      Tensor3333d dMandel_dFe = einsum< IJKL, KLMN >( dMandel_dCe, dCe_dFe );
      return { mandel, dMandel_dFe };
    }

    std::tuple< double, double > computeBetaP( const double alphaP ) const
    {
      const double beta           = hExpDelta * ( 1.0 - std::exp( -hExp * alphaP ) ) + alphaP * H;
      const double dBetaP_dAlphaP = hExpDelta * hExp * std::exp( -hExp * alphaP ) + H;
      return { beta, dBetaP_dAlphaP };
    }

    std::tuple< Eigen::VectorXd, Eigen::MatrixXd > computeResidualVectorAndTangent( const Eigen::VectorXd& X,
                                                                                    const Tensor33d&       FeTrial,
                                                                                    const double alphaPTrial ) const
    {
      const int idxA = 9;
      const int idxF = 10;
      using namespace Eigen;
      using mV9d = Eigen::Map< Eigen::Matrix< double, 9, 1 > >;
      using mM9d = Eigen::Map< Eigen::Matrix< double, 9, 9 > >;
      VectorXd R( 11 );
      MatrixXd dR_dX( 11, 11 );
      dR_dX.setZero();
      R.segment< 9 >( 0 ) = -mV9d( FeTrial.data() );
      R( 9 )              = -alphaPTrial;

      Tensor33d    Fe( X.segment( 0, 9 ).data() );
      const double dLambda = X( 10 );
      const double alphaP  = X( 9 );

      double betaP, dBetaP_dAlphaP;
      std::tie( betaP, dBetaP_dAlphaP ) = computeBetaP( alphaP );

      Tensor33d   mandelStress;
      Tensor3333d dMandel_dFe, d2f_dMandel_dMandel;
      std::tie( mandelStress, dMandel_dFe ) = computeMandelStress( Fe );

      double    f, df_dBetaP;
      Tensor33d df_dMandel;
      std::tie( f, df_dMandel, d2f_dMandel_dMandel, df_dBetaP ) = yieldFunctionFromStress( mandelStress, betaP );

      Tensor33d   dg_dMandel;
      Tensor3333d d2g_dMandel_dMandel;
      std::tie( dg_dMandel, d2g_dMandel_dMandel ) = plasticPotentialGradientAndHessianFromStress( mandelStress );

      Tensor33d   dGp = dLambda * dg_dMandel;
      Tensor33d   dFp;
      Tensor3333d ddFp_ddGp;
      std::tie( dFp, ddFp_ddGp ) = ContinuumMechanics::FiniteStrain::Plasticity::FlowIntegration::FirstOrderDerived::
        exponentialMap( dGp );

      Tensor3333d ddGp_dFe      = dLambda * einsum< ijmn, mnkL >( d2g_dMandel_dMandel, dMandel_dFe );
      Tensor33d   ddFp_ddLambda = einsum< IJKL, KL >( ddFp_ddGp, dg_dMandel );

      Tensor3333d ddFp_dFe     = einsum< iImn, mnkL >( ddFp_ddGp, ddGp_dFe );
      Tensor3333d dFeTrial_dFe = einsum< iI, IJKL >( Fe, ddFp_dFe ) +
                                 einsum< IK, JL, to_IJKL >( Spatial3D::I, transpose( Spatial3D::I % dFp ) );

      Tensor33d dFe_ddLambda = einsum< Ii, iJ >( Fe, ddFp_ddLambda );
      Tensor33d df_dFe       = einsum< IJ, IJKL >( df_dMandel, dMandel_dFe );

      R.segment< 9 >( 0 ) += mV9d( Tensor33d( einsum< iJ, JK >( Fe, dFp ) ).data() );
      R( idxA ) += ( alphaP + dLambda * df_dBetaP );
      R( idxF ) = f / G;

      dR_dX.block< 9, 9 >( 0, 0 )    = mM9d( dFeTrial_dFe.data() ).transpose();
      dR_dX.block< 1, 9 >( idxF, 0 ) = mV9d( df_dFe.data() ) / G;
      dR_dX.block< 9, 1 >( 0, idxF ) = mV9d( dFe_ddLambda.data() );
      dR_dX( idxF, idxA )            = ( df_dBetaP * dBetaP_dAlphaP ) / G;
      dR_dX( idxA, idxF )            = df_dBetaP;
      dR_dX( idxA, idxA )            = 1.0;

      return { R, dR_dX };
    }
  };

} // namespace Marmot::Materials
