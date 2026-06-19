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

#include "Marmot/DisplacementFiniteElement.h"

namespace Marmot::Elements {

  /**
   * @namespace Marmot::Elements::Hex8NaturalStabilization
   * @brief Provides helper functions and constants for the natural stabilization of an 8-node hexahedral element
   * (C3D8R).
   * @details This stabilization technique is designed to control hourglass (zero-energy) modes that arise from
   * reduced integration. The formulation is based on the work of Puso and Solberg (2006), "A stabilized nodally
   * integrated hexahedral element", IJNME. It involves adding stabilization forces and stiffness terms derived from
   * gradients of the stress field within the element.
   */
  namespace Hex8NaturalStabilization {

    /// @brief Nodal parent coordinates (xi) for an 8-node hex.
    inline constexpr double xi[8] = { -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0 };
    /// @brief Nodal parent coordinates (eta) for an 8-node hex.
    inline constexpr double eta[8] = { -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0 };
    /// @brief Nodal parent coordinates (zeta) for an 8-node hex.
    inline constexpr double zeta[8] = { -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };

    /// @brief Second derivatives of shape functions w.r.t. xi and eta, evaluated at nodes.
    constexpr double d2N_dXi2_dXi1[8] = { 0.125 * xi[0] * eta[0],
                                          0.125 * xi[1] * eta[1],
                                          0.125 * xi[2] * eta[2],
                                          0.125 * xi[3] * eta[3],
                                          0.125 * xi[4] * eta[4],
                                          0.125 * xi[5] * eta[5],
                                          0.125 * xi[6] * eta[6],
                                          0.125 * xi[7] * eta[7] };
    /// @brief Second derivatives of shape functions w.r.t. xi and zeta, evaluated at nodes.
    constexpr double d2N_dXi3_dXi1[8] = { 0.125 * xi[0] * zeta[0],
                                          0.125 * xi[1] * zeta[1],
                                          0.125 * xi[2] * zeta[2],
                                          0.125 * xi[3] * zeta[3],
                                          0.125 * xi[4] * zeta[4],
                                          0.125 * xi[5] * zeta[5],
                                          0.125 * xi[6] * zeta[6],
                                          0.125 * xi[7] * zeta[7] };
    /// @brief Second derivatives of shape functions w.r.t. eta and xi, evaluated at nodes.
    constexpr double d2N_dXi1_dXi2[8] = { 0.125 * xi[0] * eta[0],
                                          0.125 * xi[1] * eta[1],
                                          0.125 * xi[2] * eta[2],
                                          0.125 * xi[3] * eta[3],
                                          0.125 * xi[4] * eta[4],
                                          0.125 * xi[5] * eta[5],
                                          0.125 * xi[6] * eta[6],
                                          0.125 * xi[7] * eta[7] };
    /// @brief Second derivatives of shape functions w.r.t. eta and zeta, evaluated at nodes.
    constexpr double d2N_dXi3_dXi2[8] = { 0.125 * eta[0] * zeta[0],
                                          0.125 * eta[1] * zeta[1],
                                          0.125 * eta[2] * zeta[2],
                                          0.125 * eta[3] * zeta[3],
                                          0.125 * eta[4] * zeta[4],
                                          0.125 * eta[5] * zeta[5],
                                          0.125 * eta[6] * zeta[6],
                                          0.125 * eta[7] * zeta[7] };
    /// @brief Second derivatives of shape functions w.r.t. zeta and xi, evaluated at nodes.
    constexpr double d2N_dXi1_dXi3[8] = { 0.125 * xi[0] * zeta[0],
                                          0.125 * xi[1] * zeta[1],
                                          0.125 * xi[2] * zeta[2],
                                          0.125 * xi[3] * zeta[3],
                                          0.125 * xi[4] * zeta[4],
                                          0.125 * xi[5] * zeta[5],
                                          0.125 * xi[6] * zeta[6],
                                          0.125 * xi[7] * zeta[7] };
    /// @brief Second derivatives of shape functions w.r.t. zeta and eta, evaluated at nodes.
    constexpr double d2N_dXi2_dXi3[8] = { 0.125 * eta[0] * zeta[0],
                                          0.125 * eta[1] * zeta[1],
                                          0.125 * eta[2] * zeta[2],
                                          0.125 * eta[3] * zeta[3],
                                          0.125 * eta[4] * zeta[4],
                                          0.125 * eta[5] * zeta[5],
                                          0.125 * eta[6] * zeta[6],
                                          0.125 * eta[7] * zeta[7] };

    /// @brief Third derivatives of shape functions w.r.t. xi, eta, and zeta, evaluated at nodes.
    inline const Eigen::Matrix< double, 8, 1 > _d3N_dXi1_dXi2_dXi3 = { 0.125 * xi[0] * eta[0] * zeta[0],
                                                                       0.125 * xi[1] * eta[1] * zeta[1],
                                                                       0.125 * xi[2] * eta[2] * zeta[2],
                                                                       0.125 * xi[3] * eta[3] * zeta[3],
                                                                       0.125 * xi[4] * eta[4] * zeta[4],
                                                                       0.125 * xi[5] * eta[5] * zeta[5],
                                                                       0.125 * xi[6] * eta[6] * zeta[6],
                                                                       0.125 * xi[7] * eta[7] * zeta[7] };

    using Matrix3x8d    = Eigen::Matrix< double, 3, 8 >;
    using Matrix6x3d    = Eigen::Matrix< double, 6, 3 >;
    using Vector6d      = Eigen::Matrix< double, 6, 1 >;
    using JacobianSized = Eigen::Matrix< double, 3, 3 >;

    /**
     * @brief Constructs the 6x3 kinematic (B) matrix for a single node from a gradient vector.
     * @param[in] g The gradient vector (e.g., shape function derivatives).
     * @return The 6x3 B-matrix for a single node.
     */
    inline Eigen::Matrix< double, 6, 3 > makeBMatrix( const Eigen::Vector3d& g )
    {
      Eigen::Matrix< double, 6, 3 > B = Eigen::Matrix< double, 6, 3 >::Zero();
      B( 0, 0 )                       = g( 0 );
      B( 1, 1 )                       = g( 1 );
      B( 2, 2 )                       = g( 2 );
      B( 3, 0 )                       = g( 1 );
      B( 3, 1 )                       = g( 0 );
      B( 4, 0 )                       = g( 2 );
      B( 4, 2 )                       = g( 0 );
      B( 5, 1 )                       = g( 2 );
      B( 5, 2 )                       = g( 1 );
      return B;
    }

    /**
     * @brief Computes the gradient of the strain tensor with respect to the parent coordinates (dε/dξ).
     * @param[in] dU The nodal displacement increments (3x8 matrix).
     * @param[in] d2N_dXdXi Second derivatives of shape functions mapped to physical coordinates.
     * @return A 6x3 matrix where each column is the derivative of the Voigt strain vector w.r.t. one parent coordinate.
     */
    inline Matrix6x3d compute_dStrain_dXi( const Matrix3x8d& dU, const double d2N_dXdXi[8][3][3] )
    {

      Matrix6x3d ddStrain_dXi = Matrix6x3d::Zero(); // Incremental gradient

      // Assemble INCREMENTAL spatial strain gradients using dU(dim, Node)
      for ( int alpha = 0; alpha < 3; ++alpha ) {
        for ( int A = 0; A < 8; ++A ) {
          ddStrain_dXi( 0, alpha ) += d2N_dXdXi[A][0][alpha] * dU( 0, A );                                       // xx
          ddStrain_dXi( 1, alpha ) += d2N_dXdXi[A][1][alpha] * dU( 1, A );                                       // yy
          ddStrain_dXi( 2, alpha ) += d2N_dXdXi[A][2][alpha] * dU( 2, A );                                       // zz
          ddStrain_dXi( 3, alpha ) += d2N_dXdXi[A][1][alpha] * dU( 0, A ) + d2N_dXdXi[A][0][alpha] * dU( 1, A ); // xy
          ddStrain_dXi( 4, alpha ) += d2N_dXdXi[A][2][alpha] * dU( 0, A ) + d2N_dXdXi[A][0][alpha] * dU( 2, A ); // xz
          ddStrain_dXi( 5, alpha ) += d2N_dXdXi[A][2][alpha] * dU( 1, A ) + d2N_dXdXi[A][1][alpha] * dU( 2, A ); // yz
        }
      }

      return ddStrain_dXi;
    }

    /**
     * @brief Computes the second-order (hourglass) strain increment terms.
     * @param[in] dU The nodal displacement increments (3x8 matrix).
     * @param[in] dXi_dx The inverse of the Jacobian matrix (dξ/dx).
     * @return A tuple containing the three hourglass strain mode vectors (dε_12, dε_13, dε_23).
     */
    inline std::tuple< Vector6d, Vector6d, Vector6d > compute_dStrain_dXi1_dXi2_dXi3( const Matrix3x8d&    dU,
                                                                                      const JacobianSized& dXi_dx )
    {

      Eigen::Vector3d dU_dXi1_dXi2_dXi3 = dU * _d3N_dXi1_dXi2_dXi3;

      Vector6d dStrain_dXi1_dXi2, dStrain_dXi1_dXi3, dStrain_dXi2_dXi3;

      // ddeps_HG_12 relies on invJ.row(2) (j_3) - ABAQUS ORDER
      dStrain_dXi1_dXi2( 0 ) = dXi_dx( 2, 0 ) * dU_dXi1_dXi2_dXi3( 0 );                                           // xx
      dStrain_dXi1_dXi2( 1 ) = dXi_dx( 2, 1 ) * dU_dXi1_dXi2_dXi3( 1 );                                           // yy
      dStrain_dXi1_dXi2( 2 ) = dXi_dx( 2, 2 ) * dU_dXi1_dXi2_dXi3( 2 );                                           // zz
      dStrain_dXi1_dXi2( 3 ) = dXi_dx( 2, 1 ) * dU_dXi1_dXi2_dXi3( 0 ) + dXi_dx( 2, 0 ) * dU_dXi1_dXi2_dXi3( 1 ); // xy
      dStrain_dXi1_dXi2( 4 ) = dXi_dx( 2, 2 ) * dU_dXi1_dXi2_dXi3( 0 ) + dXi_dx( 2, 0 ) * dU_dXi1_dXi2_dXi3( 2 ); // xz
      dStrain_dXi1_dXi2( 5 ) = dXi_dx( 2, 2 ) * dU_dXi1_dXi2_dXi3( 1 ) + dXi_dx( 2, 1 ) * dU_dXi1_dXi2_dXi3( 2 ); // yz

      // ddeps_HG_13 relies on invJ.row(1) (j_2) - ABAQUS ORDER
      dStrain_dXi1_dXi3( 0 ) = dXi_dx( 1, 0 ) * dU_dXi1_dXi2_dXi3( 0 );
      dStrain_dXi1_dXi3( 1 ) = dXi_dx( 1, 1 ) * dU_dXi1_dXi2_dXi3( 1 );
      dStrain_dXi1_dXi3( 2 ) = dXi_dx( 1, 2 ) * dU_dXi1_dXi2_dXi3( 2 );
      dStrain_dXi1_dXi3( 3 ) = dXi_dx( 1, 1 ) * dU_dXi1_dXi2_dXi3( 0 ) + dXi_dx( 1, 0 ) * dU_dXi1_dXi2_dXi3( 1 );
      dStrain_dXi1_dXi3( 4 ) = dXi_dx( 1, 2 ) * dU_dXi1_dXi2_dXi3( 0 ) + dXi_dx( 1, 0 ) * dU_dXi1_dXi2_dXi3( 2 );
      dStrain_dXi1_dXi3( 5 ) = dXi_dx( 1, 2 ) * dU_dXi1_dXi2_dXi3( 1 ) + dXi_dx( 1, 1 ) * dU_dXi1_dXi2_dXi3( 2 );

      // ddeps_HG_23 relies on invJ.row(0) (j_1) - ABAQUS ORDER
      dStrain_dXi2_dXi3( 0 ) = dXi_dx( 0, 0 ) * dU_dXi1_dXi2_dXi3( 0 );
      dStrain_dXi2_dXi3( 1 ) = dXi_dx( 0, 1 ) * dU_dXi1_dXi2_dXi3( 1 );
      dStrain_dXi2_dXi3( 2 ) = dXi_dx( 0, 2 ) * dU_dXi1_dXi2_dXi3( 2 );
      dStrain_dXi2_dXi3( 3 ) = dXi_dx( 0, 1 ) * dU_dXi1_dXi2_dXi3( 0 ) + dXi_dx( 0, 0 ) * dU_dXi1_dXi2_dXi3( 1 );
      dStrain_dXi2_dXi3( 4 ) = dXi_dx( 0, 2 ) * dU_dXi1_dXi2_dXi3( 0 ) + dXi_dx( 0, 0 ) * dU_dXi1_dXi2_dXi3( 2 );
      dStrain_dXi2_dXi3( 5 ) = dXi_dx( 0, 2 ) * dU_dXi1_dXi2_dXi3( 1 ) + dXi_dx( 0, 1 ) * dU_dXi1_dXi2_dXi3( 2 );

      return { dStrain_dXi1_dXi2, dStrain_dXi1_dXi3, dStrain_dXi2_dXi3 };
    }

    /**
     * @brief Computes the first-order stabilization force vector.
     * @param[in] dU Nodal displacement increments (not used directly, but for consistency).
     * @param[in] d2N_dXdXi Second derivatives of shape functions mapped to physical coordinates.
     * @param[in] _integrationWeightOrder1 Integration weight for the first-order term.
     * @param[in] dStress_dXi The history variable for the first-order stress gradient.
     * @return The 3x8 nodal force matrix for the first-order stabilization.
     */
    inline Matrix3x8d computeFirstOrderStabilizationTerm(
      const Matrix3x8d&                                  dU,
      const double                                       d2N_dXdXi[8][3][3],
      const double                                       _integrationWeightOrder1,
      const Eigen::Map< Eigen::Matrix< double, 6, 3 > >& dStress_dXi )
    {

      Matrix3x8d F_stab1 = Matrix3x8d::Zero();

      // Distribute forces using the TOTAL accumulated stress gradient (_sigma_grad1)
      for ( int A = 0; A < 8; ++A ) {
        for ( int j = 0; j < 3; ++j ) {
          for ( int i = 0; i < 3; ++i ) {
            int    I           = ContinuumMechanics::VoigtNotation::IndicesToVoigtIndex[i][j];
            double contraction = 0.0;
            contraction += d2N_dXdXi[A][i][0] * dStress_dXi( I, 0 );
            contraction += d2N_dXdXi[A][i][1] * dStress_dXi( I, 1 );
            contraction += d2N_dXdXi[A][i][2] * dStress_dXi( I, 2 );
            F_stab1( j, A ) += _integrationWeightOrder1 * contraction;
          }
        }
      }

      return F_stab1;
    }

    /**
     * @brief Computes the second-order stabilization force vector.
     * @param[in] _integrationWeightOrder2 Integration weight for the second-order term.
     * @param[in] dStress_dXi1_dXi2 History variable for the (1,2) hourglass stress mode.
     * @param[in] dStress_dXi1_dXi3 History variable for the (1,3) hourglass stress mode.
     * @param[in] dStress_dXi2_dXi3 History variable for the (2,3) hourglass stress mode.
     * @param[in] dXi_dx The inverse of the Jacobian matrix (dξ/dx).
     * @return The 3x8 nodal force matrix for the second-order stabilization.
     */
    inline Matrix3x8d computeSecondOrderStabilizationTerm(
      const double                                       _integrationWeightOrder2,
      const Eigen::Map< Eigen::Matrix< double, 6, 1 > >& dStress_dXi1_dXi2,
      const Eigen::Map< Eigen::Matrix< double, 6, 1 > >& dStress_dXi1_dXi3,
      const Eigen::Map< Eigen::Matrix< double, 6, 1 > >& dStress_dXi2_dXi3,
      const JacobianSized&                               dXi_dx )
    {

      Eigen::Vector3d f_core = Eigen::Vector3d::Zero();

      // Distribute forces using the TOTAL accumulated hourglass stress gradients
      for ( int j = 0; j < 3; ++j ) {
        for ( int i = 0; i < 3; ++i ) {
          int I = ContinuumMechanics::VoigtNotation::IndicesToVoigtIndex[i][j];
          f_core( j ) +=
            // clang-format off
              dXi_dx( 2, i ) * dStress_dXi1_dXi2( I ) +
              dXi_dx( 1, i ) * dStress_dXi1_dXi3( I ) +
              dXi_dx( 0, i ) * dStress_dXi2_dXi3( I );
          // clang-format on
        }
      }
      f_core *= _integrationWeightOrder2;

      Matrix3x8d F_stab2 = f_core * _d3N_dXi1_dXi2_dXi3.transpose();

      return F_stab2;
    }

    /**
     * @brief Computes the tangent stiffness contribution from the first-order stabilization term.
     * @param[in] d2N_dXdXi Second derivatives of shape functions mapped to physical coordinates.
     * @param[in] _integrationWeightOrder1 Integration weight for the first-order term.
     * @param[in] C The consistent tangent modulus (6x6 matrix).
     * @return The 24x24 stabilization stiffness matrix for the first-order term.
     */
    inline Eigen::Matrix< double, 24, 24 > compute_dF_stab1_dQ( const double d2N_dXdXi[8][3][3],
                                                                const double _integrationWeightOrder1,
                                                                const Eigen::Matrix< double, 6, 6 >& C )
    {
      Eigen::Matrix< double, 24, 24 > K_stab1 = Eigen::Matrix< double, 24, 24 >::Zero();

      // Loop over the 3 parametric gradient directions (xi, eta, zeta)
      for ( int alpha = 0; alpha < 3; ++alpha ) {

        // Precompute the stabilization B-matrix for all 8 nodes
        Eigen::Matrix< double, 6, 3 > B_node[8];
        for ( int A = 0; A < 8; ++A ) {
          Eigen::Vector3d g( d2N_dXdXi[A][0][alpha], d2N_dXdXi[A][1][alpha], d2N_dXdXi[A][2][alpha] );
          B_node[A] = makeBMatrix( g );
        }

        // Assemble the 24x24 stiffness matrix (B^T * C * B)
        for ( int A = 0; A < 8; ++A ) {
          for ( int B = 0; B < 8; ++B ) {
            K_stab1.block< 3, 3 >( A * 3, B * 3 ) += _integrationWeightOrder1 * B_node[A].transpose() * C * B_node[B];
          }
        }
      }
      return K_stab1;
    }

    /**
     * @brief Computes the tangent stiffness contribution from the second-order stabilization term.
     * @param[in] _integrationWeightOrder2 Integration weight for the second-order term.
     * @param[in] C The consistent tangent modulus (6x6 matrix).
     * @param[in] dXi_dx The inverse of the Jacobian matrix (dξ/dx).
     * @return The 24x24 stabilization stiffness matrix for the second-order term.
     */
    inline Eigen::Matrix< double, 24, 24 > compute_dF_stab2_dQ( const double _integrationWeightOrder2,
                                                                const Eigen::Matrix< double, 6, 6 >& C,
                                                                const JacobianSized&                 dXi_dx )
    {
      Eigen::Matrix< double, 24, 24 > K_stab2 = Eigen::Matrix< double, 24, 24 >::Zero();

      // The 3 cross-derivative modes map to specific rows of the inverse Jacobian
      // Mode 12 (xi-eta) -> row 2; Mode 13 (xi-zeta) -> row 1; Mode 23 (eta-zeta) -> row 0
      const int row_map[3] = { 2, 1, 0 };

      for ( int beta = 0; beta < 3; ++beta ) {
        int             row = row_map[beta];
        Eigen::Vector3d j_vec( dXi_dx( row, 0 ), dXi_dx( row, 1 ), dXi_dx( row, 2 ) );

        Eigen::Matrix< double, 6, 3 > B_node[8];
        for ( int A = 0; A < 8; ++A ) {
          // The pseudo-gradient is the hourglass weight Gamma multiplied by the Jacobian row
          Eigen::Vector3d g = _d3N_dXi1_dXi2_dXi3( A ) * j_vec;
          B_node[A]         = makeBMatrix( g );
        }

        for ( int A = 0; A < 8; ++A ) {
          for ( int B = 0; B < 8; ++B ) {
            K_stab2.block< 3, 3 >( A * 3, B * 3 ) += _integrationWeightOrder2 * B_node[A].transpose() * C * B_node[B];
          }
        }
      }
      return K_stab2;
    }

  } // namespace Hex8NaturalStabilization

  /**
   * @class NaturallyStabilizedC3D8R
   * @brief A 3D, 8-node, reduced-integration hexahedral element with natural stabilization.
   * @details This element implements the C3D8R formulation, which uses single-point quadrature
   * to evaluate the internal forces and stiffness. To counteract the resulting hourglass (zero-energy) modes,
   * it employs a natural stabilization technique based on the work of Puso and Solberg (2006), "A stabilized
   * nodally integrated hexahedral element", IJNME.
   *
   * The stabilization adds forces and stiffness terms that are functions of the stress gradients within the element.
   * These terms are stored as history variables and updated incrementally. The element is suitable for both
   * explicit and implicit time integration schemes.
   *
   * @note This element requires 36 additional state variables per element for storing stabilization history,
   *       in addition to those required by the material model at the single integration point.
   */
  class NaturallyStabilizedC3D8R : public DisplacementFiniteElement< 3, 8 > {
    using Matrix3x8d = Eigen::Matrix< double, 3, 8 >;
    using Matrix6x3d = Eigen::Matrix< double, 6, 3 >;
    using Vector6d   = Eigen::Matrix< double, 6, 1 >;

    /// @brief User-defined bulk viscosity parameter for stabilization in explicit dynamics.
    double _bulkViscosity{ 0.06 };
    /// @brief User-defined scale factor to reduce the magnitude of stabilization forces (tuning parameter).
    double _scaleDownFactor{ 0.05 };
    /// @brief Characteristic element length for scaling the bulk viscosity.
    double _charElemLength{ 0.0 };
    /// @brief Flag to determine whether to use the deviatoric part of the tangent for stabilization.
    bool _useDeviatoricTangentForStabilization{ false };

    // --- STABILIZATION HISTORY VARIABLES MAPS ---
    /// @brief Map to the history variable for the first-order stress gradient (dσ/dξ). Size 6x3.
    Eigen::Map< Matrix6x3d > dStress_dXi;
    /// @brief Map to the history variable for the second-order stress gradient (d²σ/dξ₁dξ₂). Size 6x1.
    Eigen::Map< Voigt > dStress_dXi1_dXi2;
    /// @brief Map to the history variable for the second-order stress gradient (d²σ/dξ₁dξ₃). Size 6x1.
    Eigen::Map< Voigt > dStress_dXi1_dXi3;
    /// @brief Map to the history variable for the second-order stress gradient (d²σ/dξ₂dξ₃). Size 6x1.
    Eigen::Map< Voigt > dStress_dXi2_dXi3;

    /// @brief Inverse of the Jacobian matrix at the element center (dξ/dx).
    JacobianSized _dXi_dx;
    /// @brief Second derivatives of shape functions w.r.t. physical and parent coordinates (d²N/dxdξ).
    double _d2N_dXdXi[8][3][3];
    /// @brief Integration weight for first-order stabilization terms.
    double _integrationWeightOrder1;
    /// @brief Integration weight for second-order stabilization terms.
    double _integrationWeightOrder2;

  public:
    using Base = DisplacementFiniteElement< 3, 8 >;

    /**
     * @brief Construct a new Naturally Stabilized C3D8R element.
     * @param[in] elementID Unique element label.
     */
    NaturallyStabilizedC3D8R( int elementID )
      : Base( elementID, Marmot::FiniteElement::Quadrature::ReducedIntegration, Base::SectionType::Solid ),
        dStress_dXi( nullptr ),
        dStress_dXi1_dXi2( nullptr ),
        dStress_dXi1_dXi3( nullptr ),
        dStress_dXi2_dXi3( nullptr )
    {
      if ( this->qps.size() != 1 ) {
        throw std::invalid_argument(
          "StabilizedC3D8R must be initialized with uniform reduced integration (1 integration point)." );
      }
    }

    /**
     * @brief Precomputes geometry-related quantities for the element and its stabilization.
     * @details This override calls the base class implementation and then computes and stores
     * stabilization-specific geometric data, such as the inverse Jacobian at the center,
     * mixed-derivative shape function terms, and integration weights for the stabilization terms.
     */
    void initializeYourself() override
    {
      Base::initializeYourself();

      // Set the characteristic element length for stabilization based on the initial geometry
      const double V  = 8.0 * this->qps[0].detJ; // Physical volume of the element
      _charElemLength = std::cbrt( V );          // Characteristic length as cube root of volume

      const dNdXiSized    dNdXi0 = this->dNdXi( this->qps[0].xi );
      const JacobianSized J0     = this->Jacobian( dNdXi0 );
      _dXi_dx                    = J0.inverse();

      _integrationWeightOrder1 = V / 3.0;
      _integrationWeightOrder2 = V / 9.0;

      for ( int A = 0; A < 8; ++A ) {

        for ( int i = 0; i < 3; ++i ) {
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
     * @brief Maps the provided element state vector to stabilization history and quadrature point state.
     * @details The first 36 `double` values of the state vector are mapped to the stabilization
     * history variables (`dStress_dXi`, `dStress_dXi1_dXi2`, etc.). The remainder of the vector
     * is passed to the base class to manage the state of the single quadrature point.
     * @param[in] stateVars Pointer to the beginning of the element's state variable array.
     * @param[in] nStateVars Total number of state variables for this element.
     */
    void assignStateVars( double* stateVars, int nStateVars ) override
    {
      new ( &dStress_dXi ) Eigen::Map< Eigen::Matrix< double, 6, 3 > >( stateVars );
      new ( &dStress_dXi1_dXi2 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 18 );
      new ( &dStress_dXi1_dXi3 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 24 );
      new ( &dStress_dXi2_dXi3 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 30 );

      // Base class state variables (stress, strain, energy, etc.) are stored after the stabilization history variables
      Base::assignStateVars( stateVars + 36, nStateVars - 36 );
    }

    /**
     * @brief Returns the total number of required state variables for this element.
     * @return The sum of state variables for stabilization history (36) and the state variables
     *         required by the base class (for one quadrature point).
     */
    int getNumberOfRequiredStateVars() override
    {
      // 36 state variables for stabilization history + base class state variables
      return 36 + Base::getNumberOfRequiredStateVars();
    }

    /**
     * @brief Computes the internal force vector for explicit dynamics.
     * @details This method computes the internal force using a single integration point and adds
     * stabilization forces to control hourglass modes. It also includes a bulk viscosity term
     * for damping. The tangent stiffness is not computed.
     *
     * The stabilization history variables are updated based on the incremental displacement and the
     * material's tangent modulus.
     * @param[in] QTotal_ Total displacement vector (not used).
     * @param[in] dQ_ Incremental displacement vector.
     * @param[out] Pe_ Element internal force vector (residual).
     * @param[in] time Current time data.
     * @param[in] dT Time increment.
     * @param[out] pNewDT Suggested new time step scaling factor (e.g., if material update fails).
     */
    void computeYourselfExplicit( const double* QTotal_,
                                  const double* dQ_,
                                  double*       Pe_,
                                  const double* time,
                                  double        dT,
                                  double&       pNewDT ) override
    {
      using namespace Marmot;
      using namespace ContinuumMechanics::VoigtNotation;

      // Map inputs to Eigen vectors
      Map< const RhsSized > QTotal( QTotal_ );
      Map< const RhsSized > dQ( dQ_ );
      Map< RhsSized >       Pe( Pe_ );

      // Extract the single central quadrature point
      auto& qp = this->qps[0];

      // --- 0. CLASSICAL URI EVALUATION ---
      Voigt  dE = qp.B * dQ;
      auto&  S  = qp.managedStateVars->stress;
      CSized C_alg;

      // Evaluate the material implicitly to obtain the updated stress and algorithmic tangent C_alg
      MarmotMaterialHypoElastic::state3D  state;
      MarmotMaterialHypoElastic::timeInfo timeInfo;

      // set state info
      double elasticEnergyDensity = qp.J0xW > 0.0 ? qp.managedStateVars->elasticStrainEnergy / qp.J0xW : 0.0;
      double dissipation          = qp.J0xW > 0.0 ? qp.managedStateVars->dissipation / qp.J0xW : 0.0;
      state.stress                = S;
      state.elasticEnergyDensity  = elasticEnergyDensity;
      state.dissipation           = dissipation;
      state.stateVars             = qp.managedStateVars->materialStateVars.data();

      // set time info
      timeInfo.time = time[1];
      timeInfo.dT   = dT;
      try {
        qp.material->computeStress( state, C_alg, dE, timeInfo );
      }
      catch ( const Marmot::StressUpdateFailed& e ) {
        pNewDT = 0.5;
        // If material fails, we return early. Stabilization history is safely left unmodified.
        return;
      }

      const double density             = qp.material->getDensity( state.stateVars );
      const double dilationalWaveSpeed = qp.material->getMaximumWaveSpeed( state );

      S             = state.stress; // Updated stress after material evaluation
      auto S_damped = S;

      if ( dT >= 1e-12 ) {
        const double I1dE  = dE.head( 3 ).sum(); // Trace of the strain increment (volumetric part)
        const double p_bv1 = _bulkViscosity * density * dilationalWaveSpeed * _charElemLength * 1. / dT * ( I1dE );
        for ( int i = 0; i < 3; i++ )
          S_damped( i ) += p_bv1;
      }

      // // =====================================================================
      // // --- DEVIATORIC TANGENT PROJECTION (SYMMETRIC B-BAR) ---
      // // =====================================================================

      // Enable to use deviatoric tangent for the stabilization terms only!
      if ( _useDeviatoricTangentForStabilization ) {
        C_alg = ( IDev * C_alg ).eval();
      }

      elasticEnergyDensity = state.elasticEnergyDensity;
      dissipation          = state.dissipation;

      // Subtract the classical under-integrated internal force vector from the RHS
      Pe -= qp.B.transpose() * S_damped * qp.J0xW;

      qp.managedStateVars->elasticStrainEnergy = elasticEnergyDensity * qp.J0xW;
      qp.managedStateVars->dissipation         = dissipation * qp.J0xW;
      qp.managedStateVars->totalStrainEnergy   = ( elasticEnergyDensity + dissipation ) * qp.J0xW;
      qp.managedStateVars->strain += dE;

      // =====================================================================
      // --- GEOMETRY & KINEMATIC SETUP FOR STABILIZATION ---
      // =====================================================================

      Map< Matrix3x8d > Pe_mat( Pe.data() );

      Map< const Matrix3x8d > dU( dQ.data() );

      // =====================================================================
      // --- 1. FIRST-ORDER STABILIZATION (f_stab1) ---
      // =====================================================================

      const auto ddStrain_dXi = Hex8NaturalStabilization::compute_dStrain_dXi( dU, _d2N_dXdXi );

      Matrix6x3d ddsigma_dXi = C_alg * ddStrain_dXi;

      this->dStress_dXi += ddsigma_dXi;

      const auto F_stab1 = Hex8NaturalStabilization::computeFirstOrderStabilizationTerm( dU,
                                                                                         _d2N_dXdXi,
                                                                                         _integrationWeightOrder1,
                                                                                         dStress_dXi );

      // =====================================================================
      // --- 2. SECOND-ORDER STABILIZATION (f_stab2) ---
      // =====================================================================

      const auto [dStrain_dXi1_dXi2,
                  dStrain_dXi1_dXi3,
                  dStrain_dXi2_dXi3] = Hex8NaturalStabilization::compute_dStrain_dXi1_dXi2_dXi3( dU, _dXi_dx );

      // Project increments through tangent and update history
      this->dStress_dXi1_dXi2 += C_alg * dStrain_dXi1_dXi2;
      this->dStress_dXi1_dXi3 += C_alg * dStrain_dXi1_dXi3;
      this->dStress_dXi2_dXi3 += C_alg * dStrain_dXi2_dXi3;

      const auto F_stab2 = Hex8NaturalStabilization::computeSecondOrderStabilizationTerm( _integrationWeightOrder2,
                                                                                          dStress_dXi1_dXi2,
                                                                                          dStress_dXi1_dXi3,
                                                                                          dStress_dXi2_dXi3,
                                                                                          _dXi_dx );

      // =====================================================================
      // --- 3. FINAL FORCE ASSEMBLY ---
      // =====================================================================

      // Subtract stabilizing forces natively from the mapped 3x8 load vector
      Pe_mat -= _scaleDownFactor * ( F_stab1 + F_stab2 );
    }

    /**
     * @brief Computes the internal force vector and consistent tangent stiffness matrix for implicit analysis.
     * @details This method computes the internal force and tangent stiffness using a single integration point.
     * It then adds the stabilization forces and the corresponding consistent tangent contributions to
     * control hourglass modes.
     *
     * The stabilization history variables and their tangent contributions are updated based on the
     * incremental displacement and the material's algorithmic tangent.
     * @param[in] QTotal_ Total displacement vector (not used).
     * @param[in] dQ_ Incremental displacement vector.
     * @param[out] Pe_ Element internal force vector (residual).
     * @param[out] Ke_ Element tangent stiffness matrix.
     * @param[in] time Current time data.
     * @param[in] dT Time increment.
     * @param[out] pNewDT Suggested new time step scaling factor (e.g., if material update fails).
     */
    void computeYourself( const double* QTotal_,
                          const double* dQ_,
                          double*       Pe_,
                          double*       Ke_,
                          const double* time,
                          double        dT,
                          double&       pNewDT ) override
    {
      using namespace Marmot;
      using namespace ContinuumMechanics::VoigtNotation;

      Map< const RhsSized > QTotal( QTotal_ );
      Map< const RhsSized > dQ( dQ_ );
      Map< KeSizedMatrix >  Ke( Ke_ );
      Map< RhsSized >       Pe( Pe_ );

      Voigt  S  = Voigt::Zero();
      Voigt  dE = Voigt::Zero();
      CSized C  = CSized::Zero();

      const auto& qp = this->qps[0];

      const BSized& B             = qp.B;
      dE                          = B * dQ;
      double elasticEnergyDensity = qp.J0xW > 0.0 ? qp.managedStateVars->elasticStrainEnergy / qp.J0xW : 0.0;
      double dissipation          = qp.J0xW > 0.0 ? qp.managedStateVars->dissipation / qp.J0xW : 0.0;

      MarmotMaterialHypoElastic::state3D  state;
      MarmotMaterialHypoElastic::timeInfo timeInfo;

      // set state info
      state.stress               = qp.managedStateVars->stress;
      state.elasticEnergyDensity = elasticEnergyDensity;
      state.dissipation          = dissipation;
      state.stateVars            = qp.managedStateVars->materialStateVars.data();

      // set time info
      timeInfo.time = time[1];
      timeInfo.dT   = dT;
      try {
        qp.material->computeStress( state, C, dE, timeInfo );
      }
      catch ( const Marmot::StressUpdateFailed& e ) {
        pNewDT = 0.5;
        return;
      }

      qp.managedStateVars->stress = state.stress;
      S                           = state.stress;
      elasticEnergyDensity        = state.elasticEnergyDensity;
      dissipation                 = state.dissipation;

      qp.managedStateVars->elasticStrainEnergy = elasticEnergyDensity * qp.J0xW;
      qp.managedStateVars->dissipation         = dissipation * qp.J0xW;
      qp.managedStateVars->totalStrainEnergy   = ( elasticEnergyDensity + dissipation ) * qp.J0xW;
      qp.managedStateVars->strain += make3DVoigt< ParentGeometryElement::voigtSize >( dE );

      Ke += B.transpose() * C * B * qp.J0xW;
      Pe -= B.transpose() * S * qp.J0xW;

      // if (_useDeviatoricTangentForStabilization) {
      //   C = ( IDev * C ).eval();
      // }

      Map< const Matrix3x8d > dU( dQ.data() );

      const auto ddStrain_dXi = Hex8NaturalStabilization::compute_dStrain_dXi( dU, _d2N_dXdXi );

      Matrix6x3d ddsigma_dXi = C * ddStrain_dXi;

      this->dStress_dXi += ddsigma_dXi;

      const auto F_stab1 = Hex8NaturalStabilization::computeFirstOrderStabilizationTerm( dU,
                                                                                         _d2N_dXdXi,
                                                                                         _integrationWeightOrder1,
                                                                                         dStress_dXi );

      // =====================================================================
      // --- 2. SECOND-ORDER STABILIZATION (f_stab2) ---
      // =====================================================================

      const auto [dStrain_dXi1_dXi2,
                  dStrain_dXi1_dXi3,
                  dStrain_dXi2_dXi3] = Hex8NaturalStabilization::compute_dStrain_dXi1_dXi2_dXi3( dU, _dXi_dx );

      // Project increments through tangent and update history
      this->dStress_dXi1_dXi2 += C * dStrain_dXi1_dXi2;
      this->dStress_dXi1_dXi3 += C * dStrain_dXi1_dXi3;
      this->dStress_dXi2_dXi3 += C * dStrain_dXi2_dXi3;

      const auto F_stab2 = Hex8NaturalStabilization::computeSecondOrderStabilizationTerm( _integrationWeightOrder2,
                                                                                          dStress_dXi1_dXi2,
                                                                                          dStress_dXi1_dXi3,
                                                                                          dStress_dXi2_dXi3,
                                                                                          _dXi_dx );

      // =====================================================================
      // --- 3. FINAL FORCE ASSEMBLY ---
      // =====================================================================

      // const double secondScaleDown = exp( -1e1 * dissipation / ( std::abs( elasticEnergyDensity ) + 1e-12 ) );
      const double secondScaleDown = 1.0;

      // Subtract stabilizing forces natively from the mapped 3x8 load vector
      Pe -= _scaleDownFactor * secondScaleDown * ( F_stab1 + F_stab2 ).reshaped();

      // Add mathematically exact stabilization tangents directly to Ke
      Ke += _scaleDownFactor * secondScaleDown *
            Hex8NaturalStabilization::compute_dF_stab1_dQ( _d2N_dXdXi, _integrationWeightOrder1, C );
      Ke += _scaleDownFactor * secondScaleDown *
            Hex8NaturalStabilization::compute_dF_stab2_dQ( _integrationWeightOrder2, C, _dXi_dx );
    }
  };

} // namespace Marmot::Elements
