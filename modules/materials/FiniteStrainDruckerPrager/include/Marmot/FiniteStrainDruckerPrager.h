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

    // implementation
    /** Algorithm variant selector (read from @c materialProperties[9]). */
    const int implementationType;
    // mass properties;
    /** Density (read from @c materialProperties[10]) (if provided). */
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

    void computeStressWithScalarReturnMapping( ConstitutiveResponse< 3 >& response,
                                               AlgorithmicModuli< 3 >&    tangents,
                                               const Deformation< 3 >&    deformation,
                                               const TimeIncrement&       timeIncrement ) const;

    void computeStressWithFullReturnMapping( ConstitutiveResponse< 3 >& response,
                                             AlgorithmicModuli< 3 >&    tangents,
                                             const Deformation< 3 >&    deformation,
                                             const TimeIncrement&       timeIncrement ) const;

    void computeStressFDAF( ConstitutiveResponse< 3 >& response,
                            AlgorithmicModuli< 3 >&    tangents,
                            const Deformation< 3 >&    deformation,
                            const TimeIncrement&       timeIncrement ) const;

    void computeStressFDAC( ConstitutiveResponse< 3 >& response,
                            AlgorithmicModuli< 3 >&    tangents,
                            const Deformation< 3 >&    deformation,
                            const TimeIncrement&       timeIncrement ) const;

    void computeStressCSDA( ConstitutiveResponse< 3 >& response,
                            AlgorithmicModuli< 3 >&    tangents,
                            const Deformation< 3 >&    deformation,
                            const TimeIncrement&       timeIncrement ) const;

    double getDensity( const double* stateVars ) const override
    {
      if ( this->nMaterialProperties < 11 ) {
        throw std::runtime_error(
          std::string( MakeString() << __PRETTY_FUNCTION__ << ": No density given! nMaterialProperties < 11." ) );
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

    /** @brief Yield function templated w.r.t. Mandel stress. */
    template < typename T >
    T yieldFunctionFromStress( const Tensor33t< T >& mandelStress, const T betaP ) const
    {
      const T              sigM = 1.0 / 3.0 * trace( mandelStress );
      const Tensor33t< T > dev  = deviatoric( mandelStress );
      T                    rho  = sqrt( Fastor::inner( dev, dev ) );
      if ( double( rho ) == 0.0 )
        rho += 1e-15;
      const T q = sqrt( 1.5 ) * rho;
      const T f = q + T( Constants::sqrt3 * beta ) * sigM - betaP - T( Constants::sqrt3 * alpha_0 );
      return f;
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

    /** @brief First-order derivatives of f(M, \beta_p) w.r.t. Mandel stress (templated). */
    template < typename T >
    std::tuple< T, Tensor33t< T >, T > yieldFunctionFromStressFirstOrderDerived( const Tensor33t< T >& mandelStress,
                                                                                 const T               betaP ) const
    {
      const T        sigM = 1.0 / 3.0 * trace( mandelStress );
      Tensor33t< T > dev  = deviatoric( mandelStress );
      T              rho  = sqrt( Fastor::inner( dev, dev ) );
      if ( Math::makeReal( rho ) == 0.0 )
        rho += 1e-15;
      const T q = sqrt( 1.5 ) * rho;
      const T f = q + T( Constants::sqrt3 * beta ) * sigM - betaP - T( Constants::sqrt3 * alpha_0 );

      const Tensor33t< T > dRho_dMandel  = multiplyFastorTensorWithScalar( dev, 1. / rho );
      const Tensor33t< T > dSigM_dMandel = multiplyFastorTensorWithScalar( fastorTensorFromDoubleTensor< T >(
                                                                             Spatial3D::I ),
                                                                           T( 1. / 3. ) );
      const Tensor33t< T > df_dMandel    = multiplyFastorTensorWithScalar( dRho_dMandel, T( Constants::sqrt3_2 ) ) +
                                        multiplyFastorTensorWithScalar( dSigM_dMandel, T( Constants::sqrt3 * beta ) );

      return { f, df_dMandel, T( -1.0 ) };
    }

    /** @brief Plastic potential gradient w.r.t. Mandel stress (templated). */
    template < typename T >
    Tensor33t< T > plasticPotentialGradientFromStress( const Tensor33t< T >& mandelStress ) const
    {
      Tensor33t< T > dev = deviatoric( mandelStress );
      T              rho = sqrt( Fastor::inner( dev, dev ) );
      if ( Math::makeReal( rho ) == 0.0 )
        rho += 1e-15;

      const Tensor33t< T > dRho_dMandel  = multiplyFastorTensorWithScalar( dev, 1. / rho );
      const Tensor33t< T > dSigM_dMandel = multiplyFastorTensorWithScalar( fastorTensorFromDoubleTensor< T >(
                                                                             Spatial3D::I ),
                                                                           T( 1. / 3. ) );
      const Tensor33t< T > dg_dMandel    = multiplyFastorTensorWithScalar( dRho_dMandel, T( Constants::sqrt3_2 ) ) +
                                        multiplyFastorTensorWithScalar( dSigM_dMandel, T( Constants::sqrt3 * b ) );

      return dg_dMandel;
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

    template < typename T >
    Tensor33t< T > computeMandelStressOnly( const Tensor33t< T >& Fe ) const
    {
      using namespace Marmot::ContinuumMechanics;
      Tensor33t< T > Ce = DeformationMeasures::rightCauchyGreen( Fe );

      T              psi_;
      Tensor33t< T > dPsi_dCe;

      std::tie( psi_, dPsi_dCe )  = EnergyDensityFunctions::FirstOrderDerived::PenceGouPotentialB( Ce, K, G );
      Tensor33t< T >       PK2    = multiplyFastorTensorWithScalar( dPsi_dCe, T( 2.0 ) );
      const Tensor33t< T > mandel = Ce % PK2;
      return mandel;
    }

    template < typename T >
    T computeBetaPOnly( const T alphaP ) const
    {
      return T( hExpDelta ) * ( T( 1.0 ) - exp( -alphaP * T( hExp ) ) ) + alphaP * T( H );
    }

    std::tuple< double, double > computeBetaP( const double alphaP ) const
    {
      const double beta           = hExpDelta * ( 1.0 - std::exp( -hExp * alphaP ) ) + alphaP * H;
      const double dBetaP_dAlphaP = hExpDelta * hExp * std::exp( -hExp * alphaP ) + H;
      return { beta, dBetaP_dAlphaP };
    }

    template < typename T >
    Tensor33t< T > computeReturnMappingDirection( Tensor33t< T > Fe, T betaP ) const
    {
      Tensor33t< T > mandelStressTrial = computeMandelStressOnly( Fe );
      Tensor33t< T > dg_dS             = plasticPotentialGradientFromStress( mandelStressTrial );
      return dg_dS;
    }

    template < typename T >
    std::tuple< Tensor33d, Tensor33d > computePlasticIncrement( Tensor33t< T > dg_dS, T dLambda ) const
    {
      Tensor33d   dGp = multiplyFastorTensorWithScalar( dg_dS, dLambda );
      Tensor33d   dFp;
      Tensor3333d ddFp_ddGp;
      std::tie( dFp, ddFp_ddGp ) = ContinuumMechanics::FiniteStrain::Plasticity::FlowIntegration::FirstOrderDerived::
        exponentialMap( dGp );

      return { dFp, einsum< IJKL, KL >( ddFp_ddGp, dg_dS ) };
    }

    template < typename T >
    VectorXt< T > computeResidualVector( const VectorXt< T >& X,
                                         const Tensor33d&     FeTrial,
                                         const double         alphaPTrial ) const
    {
      const int idxA = 9;
      const int idxF = 10;
      using namespace Eigen;
      using mV9t = Eigen::Map< Eigen::Matrix< T, 9, 1 > >;
      VectorXt< T > R( 11 );
      R( 9 ) = -alphaPTrial;

      const Tensor33t< T > Fe( X.segment( 0, 9 ).data() );
      const T              dLambda = X( 10 );
      const T              alphaP  = X( 9 );
      const T              betaP   = computeBetaPOnly( alphaP );

      Tensor33t< T > mandelStress = computeMandelStressOnly( Fe );

      T              f, df_dBetaP;
      Tensor33t< T > df_dMandel;
      std::tie( f, df_dMandel, df_dBetaP ) = yieldFunctionFromStressFirstOrderDerived( mandelStress, betaP );

      Tensor33t< T >       dg_dMandel = plasticPotentialGradientFromStress( mandelStress );
      const Tensor33t< T > dGp        = multiplyFastorTensorWithScalar( dg_dMandel, dLambda );
      const Tensor33t< T > dFp = ContinuumMechanics::FiniteStrain::Plasticity::FlowIntegration::exponentialMap( dGp );

      VectorXt< T > aux = mV9t( Tensor33t< T >( einsum< iJ, JK >( Fe, dFp ) ).data() ) -
                          mV9t( fastorTensorFromDoubleTensor< T >( FeTrial ).data() );
      R( 0 )    = aux( 0 );
      R( 1 )    = aux( 1 );
      R( 2 )    = aux( 2 );
      R( 3 )    = aux( 3 );
      R( 4 )    = aux( 4 );
      R( 5 )    = aux( 5 );
      R( 6 )    = aux( 6 );
      R( 7 )    = aux( 7 );
      R( 8 )    = aux( 8 );
      R( idxA ) = ( alphaP + dLambda * df_dBetaP ) - alphaPTrial;
      R( idxF ) = f / T( G );

      return R;
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
