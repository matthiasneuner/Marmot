/* ---------------------------------------------------------------------
 *                                       _
 *  _ __ ___   __ _ _ __ _ __ ___   ___ | |_
 * | '_ ` _ \ / _` | '__| '_ ` _ \ / _ \| __|
 * | | | | | | (_| | |  | | | | | | (_) | |_
 * |_| |_| |_|\__,_|_|  |_| |_| |_|\___/ \__|
 *
 * Unit of Strength of Materials and Structural Analysis
 * University of Innsbruck,
 * 2020 - today
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

#include "Marmot/GeneralGradientEnhancedDisplacementFiniteElement.h"
#include "Marmot/NaturallyStabilizedDisplacementFiniteElement.h"

namespace Marmot::Elements {

  namespace Hex8NaturalGradientEnhancedStabilization {

    /**
     * @brief Computes the gradient of incremental nonlocal variables w.r.t. parent coordinates (dK/dxi).
     * @param[in] dqK_mat The nodal incremental nonlocal variables (8xN matrix).
     * @return An Nx3 matrix where each column is the derivative of the nonlocal variables w.r.t. one parent coordinate.
     */
    template < int nNonlocalVariables >
    inline Eigen::Matrix< double, nNonlocalVariables, 3 > compute_ddK_dxi(
      const Eigen::Matrix< double, 8, nNonlocalVariables >& dqK_mat )
    {
      using MatrixN3   = Eigen::Matrix< double, nNonlocalVariables, 3 >;
      MatrixN3 ddK_dxi = MatrixN3::Zero();
      for ( int A = 0; A < 8; ++A ) {
        double Np[3] = { 0.125 * Hex8NaturalStabilization::xi[A],
                         0.125 * Hex8NaturalStabilization::eta[A],
                         0.125 * Hex8NaturalStabilization::zeta[A] };
        for ( int alpha = 0; alpha < 3; ++alpha ) {
          for ( int n = 0; n < nNonlocalVariables; ++n ) {
            ddK_dxi( n, alpha ) += Np[alpha] * dqK_mat( A, n );
          }
        }
      }
      return ddK_dxi;
    }

    /**
     * @brief Computes the first-order nonlocal stabilization force vector.
     * @param[in] _integrationWeightOrder1 Integration weight for the first-order term.
     * @param[in] _d2N_dXdXi Second derivatives of shape functions mapped to physical coordinates.
     * @param[in] qK_mat Nodal total nonlocal variables (8xN matrix).
     * @param[in] c Nonlocal interaction parameters (Nx1 vector).
     * @return The 8xN nodal force matrix for the first-order nonlocal stabilization.
     */
    template < int nNonlocalVariables >
    inline Eigen::Matrix< double, 8, nNonlocalVariables > computeFirstOrderNonlocalStabilizationTerm(
      const double                                          _integrationWeightOrder1,
      const double                                          _d2N_dXdXi[8][3][3],
      const Eigen::Matrix< double, 8, nNonlocalVariables >& qK_mat,
      const Eigen::Matrix< double, nNonlocalVariables, 1 >& c )
    {
      using Matrix8Nd     = Eigen::Matrix< double, 8, nNonlocalVariables >;
      Matrix8Nd F_stab1_K = Matrix8Nd::Zero();
      for ( int n = 0; n < nNonlocalVariables; ++n ) {
        for ( int alpha = 0; alpha < 3; ++alpha ) {
          for ( int i = 0; i < 3; ++i ) {
            double d2K = 0.0;
            for ( int B = 0; B < 8; ++B ) {
              d2K += _d2N_dXdXi[B][i][alpha] * qK_mat( B, n );
            }
            double eta_ia = c( n ) * d2K;
            for ( int A = 0; A < 8; ++A ) {
              F_stab1_K( A, n ) += _integrationWeightOrder1 * _d2N_dXdXi[A][i][alpha] * eta_ia;
            }
          }
        }
      }
      return F_stab1_K;
    }

    /**
     * @brief Computes the second-order (hourglass) nonlocal gradient increment terms.
     * @param[in] dqK_mat The nodal incremental nonlocal variables (8xN matrix).
     * @return A tuple containing the three hourglass nonlocal mode vectors.
     */
    template < int nNonlocalVariables >
    inline std::tuple< Eigen::Matrix< double, nNonlocalVariables, 1 >,
                       Eigen::Matrix< double, nNonlocalVariables, 1 >,
                       Eigen::Matrix< double, nNonlocalVariables, 1 > >
    compute_dd2K_dxi( const Eigen::Matrix< double, 8, nNonlocalVariables >& dqK_mat )
    {
      using VectorNd   = Eigen::Matrix< double, nNonlocalVariables, 1 >;
      VectorNd dd2K_12 = VectorNd::Zero(), dd2K_13 = VectorNd::Zero(), dd2K_23 = VectorNd::Zero();
      for ( int A = 0; A < 8; ++A ) {
        double n12 = 0.125 * Hex8NaturalStabilization::xi[A] * Hex8NaturalStabilization::eta[A];
        double n13 = 0.125 * Hex8NaturalStabilization::xi[A] * Hex8NaturalStabilization::zeta[A];
        double n23 = 0.125 * Hex8NaturalStabilization::eta[A] * Hex8NaturalStabilization::zeta[A];
        for ( int n = 0; n < nNonlocalVariables; ++n ) {
          dd2K_12( n ) += n12 * dqK_mat( A, n );
          dd2K_13( n ) += n13 * dqK_mat( A, n );
          dd2K_23( n ) += n23 * dqK_mat( A, n );
        }
      }
      return { dd2K_12, dd2K_13, dd2K_23 };
    }

    /**
     * @brief Computes the second-order nonlocal stabilization force vector.
     * @param[in] _integrationWeightOrder2 Integration weight for the second-order term.
     * @param[in] qK_mat Nodal total nonlocal variables (8xN matrix).
     * @param[in] c Nonlocal interaction parameters (Nx1 vector).
     * @param[in] _dXi_dx The inverse Jacobian matrix at the element center (3x3).
     * @return The 8xN nodal force matrix for the second-order nonlocal stabilization.
     */
    template < int nNonlocalVariables >
    inline Eigen::Matrix< double, 8, nNonlocalVariables > computeSecondOrderNonlocalStabilizationTerm(
      const double                                          _integrationWeightOrder2,
      const Eigen::Matrix< double, 8, nNonlocalVariables >& qK_mat,
      const Eigen::Matrix< double, nNonlocalVariables, 1 >& c,
      const Eigen::Matrix< double, 3, 3 >&                  _dXi_dx )
    {
      using VectorNd       = Eigen::Matrix< double, nNonlocalVariables, 1 >;
      using Matrix8Nd      = Eigen::Matrix< double, 8, nNonlocalVariables >;
      VectorNd k_Gamma     = qK_mat.transpose() * Hex8NaturalStabilization::_d3N_dXi1_dXi2_dXi3;
      VectorNd f_core_K    = VectorNd::Zero();
      double   invJ_sq_sum = 0.0;
      for ( int i = 0; i < 3; ++i ) {
        invJ_sq_sum += _dXi_dx( 2, i ) * _dXi_dx( 2, i ) + _dXi_dx( 1, i ) * _dXi_dx( 1, i ) +
                       _dXi_dx( 0, i ) * _dXi_dx( 0, i );
      }
      for ( int n = 0; n < nNonlocalVariables; ++n ) {
        f_core_K( n ) = _integrationWeightOrder2 * c( n ) * k_Gamma( n ) * invJ_sq_sum;
      }
      Matrix8Nd F_stab2_K = Hex8NaturalStabilization::_d3N_dXi1_dXi2_dXi3 * f_core_K.transpose();
      return F_stab2_K;
    }

  } // namespace Hex8NaturalGradientEnhancedStabilization

  /**
   * @brief A stabilized gradient-enhanced hexahedral element (C3D8R) for finite strain mechanics.
   *
   * This class implements a stabilized version of the gradient-enhanced displacement finite element,
   * specifically for a hexahedral element with reduced integration (GC3D8R).
   * It incorporates stabilization techniques to mitigate spurious oscillations, particularly in
   * nearly incompressible or poorly conditioned problems.
   *
   * @tparam nNonlocalVariables The number of nonlocal variables used for stabilization.
   */
  template < int nNonlocalVariables = 1 >
  class NaturallyStabilizedGradientEnhancedC3D8R
    : public GeneralGradientEnhancedDisplacementFiniteElement< 3, 8, nNonlocalVariables, 8 > {

    using Base = GeneralGradientEnhancedDisplacementFiniteElement< 3, 8, nNonlocalVariables, 8 >;

    /**
     * @brief User-defined bulk viscosity parameter for stabilization.
     * This parameter controls the strength of the artificial viscosity added for stabilization.
     */
    double _bulkViscosity{ 0.06 };
    /**
     * @brief Characteristic element length for scaling the bulk viscosity.
     * This is derived from the element's volume and is used to appropriately scale stabilization terms.
     */
    double _charElemLength{ 0.0 };

    // ---------------------------------------------------------------------
    // STABILIZATION HISTORY VARIABLES MAPS
    // These Eigen maps provide convenient access to the memory allocated for
    // stabilization history variables.
    /**
     * @brief Mapped state variable for the derivative of stress with respect to local element coordinates (xi, eta,
     * zeta). Stores d(sigma)/d(xi), d(sigma)/d(eta), d(sigma)/d(zeta).
     */
    Eigen::Map< Eigen::Matrix< double, 6, 3 > > dStress_dXi;
    /**
     * @brief Mapped state variable for stress derivatives related to mixed second-order gradients.
     * Stores terms like d(sigma)/d(xi)d(eta).
     */
    Eigen::Map< Eigen::Matrix< double, 6, 1 > > dStress_dXi1_dXi2;
    /**
     * @brief Mapped state variable for stress derivatives related to mixed second-order gradients.
     * Stores terms like d(sigma)/d(xi)d(zeta).
     */
    Eigen::Map< Eigen::Matrix< double, 6, 1 > > dStress_dXi1_dXi3;
    /**
     * @brief Mapped state variable for stress derivatives related to mixed second-order gradients.
     * Stores terms like d(sigma)/d(eta)d(zeta).
     */
    Eigen::Map< Eigen::Matrix< double, 6, 1 > > dStress_dXi2_dXi3;

    /**
     * @brief Inverse Jacobian matrix mapping from local to global coordinates.
     * Stores dx/dXi.
     */
    typename Base::JacobianSized _dXi_dx;
    /**
     * @brief Second derivatives of shape functions with respect to global Cartesian coordinates.
     * Stores d^2N/dx_i dx_j for each node and each coordinate direction.
     */
    double _d2N_dXdXi[8][3][3];
    /**
     * @brief Integration weight for terms involving first-order derivatives.
     * Used in the first-order stabilization terms.
     */
    double _integrationWeightOrder1;
    /**
     * @brief Integration weight for terms involving second-order derivatives.
     * Used in the second-order stabilization terms.
     */
    double _integrationWeightOrder2;

  public:
    /**
     * @brief Constructor for the NaturallyStabilizedGradientEnhancedC3D8R element.
     *
     * @param elementID The unique identifier for this element.
     * @throws std::invalid_argument If the element is not initialized with uniform reduced integration (1 integration
     * point).
     */
    NaturallyStabilizedGradientEnhancedC3D8R( int elementID )
      : Base( elementID, Marmot::FiniteElement::Quadrature::ReducedIntegration, Base::SectionType::Solid ),
        dStress_dXi( nullptr ),
        dStress_dXi1_dXi2( nullptr ),
        dStress_dXi1_dXi3( nullptr ),
        dStress_dXi2_dXi3( nullptr )
    {
      if ( this->qps.size() != 1 ) {
        throw std::invalid_argument(
          "Stabilized GC3D8R must be initialized with uniform reduced integration (1 integration point)." );
      }
    }

    /**
     * @brief Initializes element-specific data structures.
     *
     * This method computes the characteristic element length, the inverse Jacobian matrix,
     * and the second derivatives of the shape functions with respect to the global
     * Cartesian coordinates. These are crucial for stabilization calculations.
     */
    void initializeYourself() override
    {
      Base::initializeYourself();

      // Set the characteristic element length for stabilization based on the initial geometry
      const double V  = 8.0 * this->qps[0].detJ; // Volume of the element
      _charElemLength = std::cbrt( V );

      // Calculate the inverse Jacobian at the quadrature point
      const typename Base::dNdXiSized    dNdXi0 = this->localGeometryElement.dNdXi( this->qps[0].xi );
      const typename Base::JacobianSized J0     = this->localGeometryElement.Jacobian( dNdXi0 );
      _dXi_dx                                   = J0.inverse();

      // Calculate integration weights based on element volume
      _integrationWeightOrder1 = V / 3.0;
      _integrationWeightOrder2 = V / 9.0;

      // Compute second derivatives of shape functions w.r.t global coordinates
      for ( int A = 0; A < 8; ++A ) {   // Loop over nodes
        for ( int i = 0; i < 3; ++i ) { // Loop over global coordinate axes (x, y, z)
          // Compute d^2N_A / dx_i dx_j by applying chain rule with _dXi_dx
          _d2N_dXdXi[A][i][0] = Hex8NaturalStabilization::d2N_dXi2_dXi1[A] * _dXi_dx( 1, i ) +
                                Hex8NaturalStabilization::d2N_dXi3_dXi1[A] * _dXi_dx( 2, i );
          _d2N_dXdXi[A][i][1] = Hex8NaturalStabilization::d2N_dXi1_dXi2[A] * _dXi_dx( 0, i ) +
                                Hex8NaturalStabilization::d2N_dXi3_dXi2[A] * _dXi_dx( 2, i );
          _d2N_dXdXi[A][i][2] = Hex8NaturalStabilization::d2N_dXi1_dXi3[A] * _dXi_dx( 0, i ) +
                                Hex8NaturalStabilization::d2N_dXi2_dXi3[A] * _dXi_dx( 1, i );
        }
      }
    }

    /**
     * @brief Assigns state variables from a raw array to mapped Eigen objects.
     *
     * This method is crucial for managing element state, including stabilization history
     * variables, which are mapped to Eigen matrices for efficient access.
     *
     * @param stateVars Pointer to the beginning of the state variables array.
     * @param nStateVars The total number of state variables for this element.
     */
    void assignStateVars( double* stateVars, int nStateVars ) override
    {
      // Map the stabilization history variables
      new ( &dStress_dXi ) Eigen::Map< Eigen::Matrix< double, 6, 3 > >( stateVars );
      new ( &dStress_dXi1_dXi2 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 18 );
      new ( &dStress_dXi1_dXi3 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 24 );
      new ( &dStress_dXi2_dXi3 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 30 );

      // Base class state variables are stored after the 36 stabilization history variables
      Base::assignStateVars( stateVars + 36, nStateVars - 36 );
    }

    /**
     * @brief Returns the total number of state variables required by the element.
     *
     * This includes the state variables managed by the base class plus the stabilization
     * history variables.
     *
     * @return The total number of state variables.
     */
    int getNumberOfRequiredStateVars() override { return 36 + Base::getNumberOfRequiredStateVars(); }

    /**
     * @brief Computes the explicit internal forces and updates element state.
     *
     * This is the core computation routine for the element. It involves:
     * 1. Evaluating material response based on current kinematics and state.
     * 2. Applying artificial bulk viscosity for stabilization.
     * 3. Accumulating baseline internal forces (mechanical and nonlocal).
     * 4. Calculating and applying first-order stabilization terms.
     * 5. Calculating and applying second-order stabilization terms (hourglass stabilization).
     * 6. Assembling the final internal forces.
     * 7. Updating the element's state variables (stress, strain, energy).
     *
     * @param QTotal_ Pointer to the array of total nodal degrees of freedom (displacements and nonlocal variables).
     * @param dQ_ Pointer to the array of incremental nodal degrees of freedom.
     * @param Pe_ Pointer to the array where the internal forces (residual) will be stored.
     * @param time Pointer to the current time information (typically time[0]=current, time[1]=previous).
     * @param dT The time increment.
     * @param pNewDT Reference to a double to potentially update the time step size (e.g., in case of material failure).
     */
    void computeYourselfExplicit( const double* QTotal_,
                                  const double* dQ_,
                                  double*       Pe_,
                                  const double* time,
                                  double        dT,
                                  double&       pNewDT )
    {
      using namespace Marmot;
      using namespace ContinuumMechanics::VoigtNotation;

      // Define response types from the material model for easier access
      using response  = typename MarmotMaterialGeneralGradientEnhancedHypoElastic< nNonlocalVariables >::response;
      using tangents  = typename MarmotMaterialGeneralGradientEnhancedHypoElastic< nNonlocalVariables >::tangents;
      using increment = typename MarmotMaterialGeneralGradientEnhancedHypoElastic< nNonlocalVariables >::increment;

      // -----------------------------------------------------------------------
      // --- 0. MEMORY MAPPING AND BASE CLASSICAL EVALUATION ---
      // -----------------------------------------------------------------------

      // Map the input arrays to Eigen objects for easier manipulation.
      Map< const typename Base::RhsSized > QTotal( QTotal_ ); // Total DOFs
      Map< const typename Base::RhsSized > dQ( dQ_ );         // Incremental DOFs
      Map< typename Base::RhsSized >       Pe( Pe_ );         // Internal forces (residual)

      // Map mechanical DOFs (3 spatial dimensions x 8 nodes)
      using Matrix3x8d = Eigen::Matrix< double, 3, 8 >; // Rows: 3 (x,y,z), Cols: 8 (nodes)
      Map< const Matrix3x8d > dU( dQ.data() );          // Incremental displacements
      Map< Matrix3x8d >       Pe_U_mat( Pe.data() );    // Mechanical part of internal forces

      // Map nonlocal DOFs (8 nodes x nNonlocalVariables)
      using Matrix8Nd = Eigen::Matrix< double, 8, nNonlocalVariables >; // Rows: 8 (nodes), Cols: nNonlocalVariables
      Map< const Matrix8Nd > dqK_mat( dQ.data() + Base::sizeDoFU );     // Incremental nonlocal variables
      Map< const Matrix8Nd > qK_mat( QTotal.data() +
                                     Base::sizeDoFU ); // Total nonlocal variables (needed for linear PDE stabilization)
      Map< Matrix8Nd >       Pe_K_mat( Pe.data() + Base::sizeDoFU ); // Nonlocal part of internal forces

      // Extract information for the current quadrature point (using reduced integration, only one qp)
      auto&                         qp  = this->qps[0];
      const typename Base::BSized&  B   = qp.B;   // Strain-displacement matrix
      const typename Base::NSizedK& N_K = qp.N_K; // Shape functions for nonlocal variables
      const typename Base::dNdXiSizedK&
        dNdX_K = qp.dNdX_K;                       // Gradient of nonlocal shape functions w.r.t global coordinates

      // Compute incremental strain vector (Voigt notation)
      typename Base::Voigt dE = B * dQ.head( Base::sizeDoFU );

      // Extract nonlocal variables K and their increments dK
      Vector< double, nNonlocalVariables > K;
      Vector< double, nNonlocalVariables > dK;
      for ( size_t n = 0; n < nNonlocalVariables; n++ ) {
        K( n )  = N_K * qK_mat.col( n );  // Total nonlocal variable n
        dK( n ) = N_K * dqK_mat.col( n ); // Incremental nonlocal variable n
      }

      // Structures to hold material response and tangent information
      response  res; // Material response (stress, energy, etc.)
      tangents  tan; // Material tangents (needed for algorithmic stabilization)
      increment inc; // Kinematic increment for material computation

      try {
        // Initialize response structure with current state variables from the quadrature point
        res.stress               = qp.managedStateVars->stress;
        res.elasticEnergyDensity = qp.managedStateVars->elasticStrainEnergy / qp.J0xW;
        res.dissipation          = qp.managedStateVars->dissipation / qp.J0xW;
        res.stateVars = qp.managedStateVars->materialStateVars.data(); // Pointer to material-specific state variables

        // Define the kinematic increment for the material model
        inc = { dE, K, dK, time[1], dT }; // dE: strain increment, K: total nonlocal, dK: nonlocal increment, time[1]:
                                          // previous time, dT: time step

        // --- Material Update ---
        qp.material->computeStress( res, tan, inc ); // Compute stress and tangents from material model

        // --- Add Bulk Viscosity Stabilization ---
        if ( dT >= 1e-12 ) { // Only apply if time step is positive
          const double density             = qp.material->getDensity( res.stateVars ); // Material density
          const double dilationalWaveSpeed = qp.material->getMaximumWaveSpeed( res );  // Max wave speed for material
          const double I1dE = dE.head( 3 ).sum(); // Sum of first three components of strain increment (dilatation)
          // Calculate artificial bulk viscosity term
          const double p_bv1 = _bulkViscosity * density * dilationalWaveSpeed * _charElemLength * 1. / dT * ( I1dE );
          // Add to normal stresses (for isotropic stress)
          for ( int i = 0; i < 3; i++ ) {
            res.stress( i ) += p_bv1;
          }
        }

        // --- Accumulate baseline internal forces ---
        // Mechanical part: Pe_U_mat = -B^T * stress * J0*W
        Pe.head( Base::sizeDoFU ) -= B.transpose() * res.stress * qp.J0xW;

        // Nonlocal part: Pe_K_mat = -( N_K^T * K + c * dNdX_K^T * dNdX_K * qK_mat - N_K^T * K_local ) * J0*W
        for ( int n = 0; n < nNonlocalVariables; n++ ) {
          Pe_K_mat.col( n ) -= ( N_K.transpose() * K( n ) + res.c( n ) * dNdX_K.transpose() * dNdX_K * qK_mat.col( n ) -
                                 N_K.transpose() * res.KLocal( n ) ) *
                               qp.J0xW;
        }
      }
      catch ( Marmot::StressUpdateFailed& e ) { // Handle material failure
        pNewDT = 0.25;                          // Reduce time step size
        // Material failure -> exit before updating any stabilization histories
        return;
      }

      // --- Update state variables in the quadrature point ---
      qp.managedStateVars->stress = res.stress; // Update stress
      // Update total strain (Voigt notation)
      qp.managedStateVars->strain += make3DVoigt< Base::ParentGeometryElement::voigtSize >( dE );
      // Update energy densities (scaled by J0*W, which will be divided out later if needed)
      qp.managedStateVars->elasticStrainEnergy = res.elasticEnergyDensity * qp.J0xW;
      qp.managedStateVars->dissipation         = res.dissipation * qp.J0xW;
      qp.managedStateVars->totalStrainEnergy   = ( res.elasticEnergyDensity + res.dissipation ) * qp.J0xW;

      // -----------------------------------------------------------------------
      // --- 1. FIRST-ORDER STABILIZATION ---
      // -----------------------------------------------------------------------

      // [A] INCREMENTAL Mechanical Kinematics: Compute gradient of incremental displacement w.r.t. local coordinates.
      // ddStrain_dXi stores d(dU)/d(xi), d(dU)/d(eta), d(dU)/d(zeta) in a structured way.
      using Matrix6x3d        = Eigen::Matrix< double, 6, 3 >; // Strain components x spatial gradients
      const auto ddStrain_dXi = Hex8NaturalStabilization::compute_dStrain_dXi( dU, _d2N_dXdXi );

      // [B] INCREMENTAL Nonlocal Kinematics: Compute gradient of incremental nonlocal variables w.r.t. local
      // coordinates.
      const auto ddK_dxi = Hex8NaturalGradientEnhancedStabilization::compute_ddK_dxi< nNonlocalVariables >( dqK_mat );

      // [C] Project through coupled tangents and accumulate history
      // Compute the change in stress due to both mechanical and nonlocal increments.
      Matrix6x3d dDeltaStress_dXi = tan.dStressddStrain * ddStrain_dXi + tan.dStressddK * ddK_dxi;
      // Accumulate this change into the history variable dStress_dXi.
      this->dStress_dXi += dDeltaStress_dXi;

      // Compute the first-order stabilization force contribution from mechanical DOFs (dU).
      const auto F_stab1_U = Hex8NaturalStabilization::computeFirstOrderStabilizationTerm( dU,
                                                                                           _d2N_dXdXi,
                                                                                           _integrationWeightOrder1,
                                                                                           dStress_dXi );

      // Compute the first-order stabilization force contribution for nonlocal DOFs.
      const auto F_stab1_K = Hex8NaturalGradientEnhancedStabilization::computeFirstOrderNonlocalStabilizationTerm<
        nNonlocalVariables >( _integrationWeightOrder1, _d2N_dXdXi, qK_mat, res.c );

      // -----------------------------------------------------------------------
      // --- 2. SECOND-ORDER STABILIZATION ---
      // -----------------------------------------------------------------------

      // [A] INCREMENTAL Mechanical Hourglass Strains: Compute second-order gradients of incremental displacements.
      // These terms capture strain variations related to hourglassing modes.
      const auto [dStrain_dXi1_dXi2, // d(strain)/d(xi)d(eta)
                  dStrain_dXi1_dXi3, // d(strain)/d(xi)d(zeta)
                  dStrain_dXi2_dXi3] = Hex8NaturalStabilization::compute_dStrain_dXi1_dXi2_dXi3( dU, _dXi_dx );

      // [B] INCREMENTAL Nonlocal Hourglass Parameters: Compute second-order mixed derivatives of incremental nonlocal
      // variables.
      const auto [dd2K_12, dd2K_13, dd2K_23] = Hex8NaturalGradientEnhancedStabilization::compute_dd2K_dxi<
        nNonlocalVariables >( dqK_mat );

      // [C] Project through coupled tangents and update history
      // Compute the change in stress due to mixed second-order increments (mechanical and nonlocal).
      // These updates are added to the mapped history variables.
      this->dStress_dXi1_dXi2 += tan.dStressddStrain * dStrain_dXi1_dXi2 + tan.dStressddK * dd2K_12;
      this->dStress_dXi1_dXi3 += tan.dStressddStrain * dStrain_dXi1_dXi3 + tan.dStressddK * dd2K_13;
      this->dStress_dXi2_dXi3 += tan.dStressddStrain * dStrain_dXi2_dXi3 + tan.dStressddK * dd2K_23;

      // Compute the second-order stabilization force contribution from mechanical DOFs (dU).
      const auto F_stab2_U = Hex8NaturalStabilization::computeSecondOrderStabilizationTerm( _integrationWeightOrder2,
                                                                                            dStress_dXi1_dXi2,
                                                                                            dStress_dXi1_dXi3,
                                                                                            dStress_dXi2_dXi3,
                                                                                            _dXi_dx );

      // Compute the second-order stabilization force contribution for nonlocal DOFs.
      const auto F_stab2_K = Hex8NaturalGradientEnhancedStabilization::computeSecondOrderNonlocalStabilizationTerm<
        nNonlocalVariables >( _integrationWeightOrder2, qK_mat, res.c, _dXi_dx );

      // -----------------------------------------------------------------------
      // --- 3. FINAL FORCE ASSEMBLY ---
      // -----------------------------------------------------------------------

      // Subtract the stabilization forces from the mechanical and nonlocal internal force vectors.
      Pe_U_mat -= ( F_stab1_U + F_stab2_U );
      Pe_K_mat -= ( F_stab1_K + F_stab2_K );
    }
  };

} // namespace Marmot::Elements
