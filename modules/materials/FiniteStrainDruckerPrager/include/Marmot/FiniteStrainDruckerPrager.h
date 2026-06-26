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
   *
   * @par Mathematical Formulation
   * The yield function @f$ f(\boldsymbol{M}, \beta_p) @f$ is formulated in Mandel stress space as:
   * @f[
   *   f(\boldsymbol{M}, \beta_p) = q + \sqrt{3}\beta \sigma_m - \beta_p - \sqrt{3}\alpha_0 \le 0
   * @f]
   * where:
   * - @f$ \boldsymbol{M} = \boldsymbol{C}^e \cdot \boldsymbol{S} @f$ is the Mandel stress tensor, with @f$
   * \boldsymbol{S} = 2 \partial \Psi / \partial \boldsymbol{C}^e @f$ as the second Piola-Kirchhoff stress.
   * - @f$ q = \sqrt{\frac{3}{2}} \rho @f$ is the equivalent von Mises stress.
   * - @f$ \rho = \sqrt{\text{dev}(\boldsymbol{M}) : \text{dev}(\boldsymbol{M})} @f$ is the norm of the deviatoric
   * Mandel stress tensor.
   * - @f$ \sigma_m = \frac{1}{3} \text{tr}(\boldsymbol{M}) @f$ is the mean stress.
   * - @f$ \beta_p @f$ is the stress-like isotropic hardening variable.
   *
   * The non-associated plastic flow potential @f$ g(\boldsymbol{M}) @f$ is defined as:
   * @f[
   *   g(\boldsymbol{M}) = q + \sqrt{3} b \sigma_m - \text{constant}
   * @f]
   *
   * The parameters @f$ \alpha_0 @f$, @f$ \beta @f$, and @f$ b @f$ are derived from cohesion @f$ c @f$, friction angle
   * @f$ \phi @f$, and dilation angle @f$ \psi @f$:
   * @f[
   *   \alpha_0 = \frac{6 c \cos\phi}{\sqrt{3}(3 + A\sin\phi)}
   * @f]
   * @f[
   *   \beta = \frac{6 \sin\phi}{\sqrt{3}(3 + A\sin\phi)}
   * @f]
   * @f[
   *   b = \frac{6 \sin\psi}{\sqrt{3}(3 + A\sin\psi)}
   * @f]
   * where @f$ A @f$ is the vertex scaling parameter.
   *
   * Isotropic hardening is governed by:
   * @f[
   *   \beta_p(\alpha_p) = h_{\text{ExpDelta}} (1 - e^{-h_{\text{Exp}} \alpha_p}) + H \alpha_p
   * @f]
   * where @f$ \alpha_p @f$ is the equivalent plastic strain-like internal variable.
   */
  class FiniteStrainDruckerPrager : public MarmotMaterialFiniteStrain {
  public:
    using MarmotMaterialFiniteStrain::MarmotMaterialFiniteStrain;

    // elastic constants
    /** @brief Bulk modulus @f$ K @f$ (read from @c materialProperties[0]). */
    const double K;
    /** @brief Shear modulus @f$ G @f$ (read from @c materialProperties[1]). */
    const double G;

    // plasticity parameters
    /** @brief Cohesion @f$ c @f$ (read from @c materialProperties[2]). */
    const double cohesion;
    /** @brief Friction angle @f$ \phi @f$ in degrees (read from @c materialProperties[3]). */
    const double phi;
    /** @brief Dilation angle @f$ \psi @f$ in degrees (read from @c materialProperties[4]). */
    const double psi;
    /** @brief Scaling factor @f$ A @f$ (read from @c materialProperties[5]). */
    const double A;
    /** @brief Hardening exponential delta @f$ h_{\text{ExpDelta}} @f$ (read from @c materialProperties[6]). */
    const double hExpDelta;
    /** @brief Hardening exponential rate @f$ h_{\text{Exp}} @f$ (read from @c materialProperties[7]). */
    const double hExp;
    /** @brief Linear hardening modulus @f$ H @f$ (read from @c materialProperties[8]). */
    const double H;

    // dependent parameters
    /** @brief Friction angle in radians @f$ \phi_{\text{rad}} @f$. */
    const double phiRad;
    /** @brief Dilation angle in radians @f$ \psi_{\text{rad}} @f$. */
    const double psiRad;
    /** @brief Cohesion parameter coefficient @f$ \alpha_0 @f$. */
    const double alpha_0;
    /** @brief Friction angle coefficient @f$ \beta @f$. */
    const double beta;
    /** @brief Dilation angle coefficient @f$ b @f$. */
    const double b;

    // mass properties;
    /** @brief Material density @f$ \rho_0 @f$ (read from @c materialProperties[9]) (if provided). */
    const double density;

    /**
     * @brief Construct the finite-strain Drucker-Prager plasticity model.
     * @param materialProperties Pointer to the array of material parameters.
     * @param nMaterialProperties Length of the material parameter array.
     * @param materialLabel Material label.
     */
    FiniteStrainDruckerPrager( const double* materialProperties, int nMaterialProperties, int materialLabel );

    /**
     * @brief Compute the Kirchhoff stress and the algorithmic tangent.
     * @details Performs an elastic trial update, evaluates the yield function, and, if required,
     * triggers the backward Euler return mapping algorithm using analytical derivatives.
     * @param[in,out] response Output stress state and elastic energy density.
     * @param[out] tangents Algorithmic tangent moduli tensor @f$ \partial \boldsymbol{\tau}/\partial \boldsymbol{F}
     * @f$.
     * @param[in] deformation Current deformation gradient @f$ \boldsymbol{F} @f$.
     * @param[in] timeIncrement Time increment @f$ \Delta t @f$.
     */
    void computeStress( ConstitutiveResponse< 3 >& response,
                        AlgorithmicModuli< 3 >&    tangents,
                        const Deformation< 3 >&    deformation,
                        const TimeIncrement&       timeIncrement ) const override;

    /**
     * @brief Retrieve the material density.
     * @param stateVars Pointer to the state variables.
     * @return Density value.
     */
    double getDensity( const double* stateVars ) const override
    {
      if ( this->nMaterialProperties < 10 ) {
        throw std::runtime_error(
          std::string( MakeString() << __PRETTY_FUNCTION__ << ": No density given! nMaterialProperties < 10." ) );
      }
      return this->density;
    }

    /**
     * @brief Initialize internal state variables.
     * @details Sets the plastic deformation gradient @f$ \boldsymbol{F}^p = \boldsymbol{I} @f$
     * and the equivalent plastic strain @f$ \alpha_p = 0 @f$.
     * @param[out] stateVars Pointer to the state variable array.
     * @param[in] nStateVars Length of the state variable array.
     */
    void initializeYourself( double* stateVars, int nStateVars ) override;

    /**
     * @brief Compute the yield function value and its first derivatives.
     * @param Fe Elastic deformation gradient @f$ \boldsymbol{F}^e @f$.
     * @param betaP Hardening variable value @f$ \beta_p @f$.
     * @return A tuple containing the yield function value @f$ f @f$, derivative @f$ \partial f / \partial
     * \boldsymbol{F}^e @f$, and derivative w.r.t hardening @f$ \partial f / \partial \beta_p = -1.0 @f$.
     */
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

    /**
     * @brief Compute the yield function value and its first and second derivatives w.r.t. Mandel stress.
     * @param mandelStress Mandel stress tensor @f$ \boldsymbol{M} @f$.
     * @param betaP Hardening variable value @f$ \beta_p @f$.
     * @return A tuple containing @f$ f @f$, gradient @f$ \partial f / \partial \boldsymbol{M} @f$,
     * Hessian @f$ \partial^2 f / \partial \boldsymbol{M} \partial \boldsymbol{M} @f$, and hardening derivative @f$
     * \partial f / \partial \beta_p = -1.0 @f$.
     */
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

    /**
     * @brief Compute the plastic potential gradient and Hessian w.r.t. Mandel stress.
     * @param mandelStress Mandel stress tensor @f$ \boldsymbol{M} @f$.
     * @return A tuple containing gradient @f$ \partial g / \partial \boldsymbol{M} @f$
     * and Hessian @f$ \partial^2 g / \partial \boldsymbol{M} \partial \boldsymbol{M} @f$.
     */
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

    /**
     * @brief Determine if the elastic trial state violates the yield condition.
     * @param Fe Elastic deformation gradient @f$ \boldsymbol{F}^e @f$.
     * @param betaP Hardening variable value @f$ \beta_p @f$.
     * @return True if @f$ f(\boldsymbol{M}, \beta_p) > 0 @f$.
     */
    bool isYielding( const Tensor33d& Fe, const double betaP ) const
    {
      double    f, df_dBetaP;
      Tensor33d df_dFe;
      std::tie( f, df_dFe, df_dBetaP ) = yieldFunction( Fe, betaP );
      return ( f > 0.0 );
    }

    /**
     * @brief Compute the Mandel stress tensor and its derivative w.r.t. the elastic deformation gradient.
     * @param Fe Elastic deformation gradient @f$ \boldsymbol{F}^e @f$.
     * @return A tuple containing Mandel stress @f$ \boldsymbol{M} @f$ and derivative @f$ \partial \boldsymbol{M} /
     * \partial \boldsymbol{F}^e @f$.
     */
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

    /**
     * @brief Compute the isotropic hardening variable and its derivative w.r.t equivalent plastic strain.
     * @param alphaP Equivalent plastic strain @f$ \alpha_p @f$.
     * @return A tuple containing @f$ \beta_p(\alpha_p) @f$ and the derivative @f$ \partial \beta_p / \partial \alpha_p
     * @f$.
     */
    std::tuple< double, double > computeBetaP( const double alphaP ) const
    {
      const double beta           = hExpDelta * ( 1.0 - std::exp( -hExp * alphaP ) ) + alphaP * H;
      const double dBetaP_dAlphaP = hExpDelta * hExp * std::exp( -hExp * alphaP ) + H;
      return { beta, dBetaP_dAlphaP };
    }

    /**
     * @brief Compute the residual vector and its Jacobian tangent matrix for the backward Euler integration.
     * @details Computes the residual vector @f$ \boldsymbol{R} @f$ and the tangent @f$ \partial \boldsymbol{R}/\partial
     * \boldsymbol{X} @f$ for the Newton-Raphson method. The system of equations incorporates:
     * - @f$ \boldsymbol{R}_{F^e} = \boldsymbol{F}^e \Delta\boldsymbol{F}^p - \boldsymbol{F}^e_{\text{trial}} =
     * \boldsymbol{0} @f$ (9 equations)
     * - @f$ R_{\alpha_p} = \alpha_p - \Delta\lambda - \alpha_{p,\text{old}} = 0 @f$ (1 equation)
     * - @f$ R_f = f(\boldsymbol{M}, \beta_p) / G = 0 @f$ (1 equation, normalized by shear modulus @f$ G @f$ to ensure
     * robust convergence)
     *
     * @param X Current iteration solution vector containing @f$ [\boldsymbol{F}^e, \alpha_p, \Delta\lambda] @f$.
     * @param FeTrial Trial elastic deformation gradient @f$ \boldsymbol{F}^e_{\text{trial}} @f$.
     * @param alphaPTrial Old value of the equivalent plastic strain @f$ \alpha_{p,\text{old}} @f$.
     * @return A tuple containing the residual vector @f$ \boldsymbol{R} @f$ and its Jacobian tangent @f$ \partial
     * \boldsymbol{R} / \partial \boldsymbol{X} @f$.
     */
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
