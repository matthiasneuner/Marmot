/* ---------------------------------------------------------------------
 * Stabilized Hex8 Element (GCD8R - Classical Continuum)
 * ---------------------------------------------------------------------
 */
#pragma once

#include "Marmot/DisplacementFiniteElement.h"

namespace Marmot::Elements {

  class NaturallyStabilizedC3D8R : public DisplacementFiniteElement< 3, 8 > {
  public:
    using Base = DisplacementFiniteElement< 3, 8 >;

    NaturallyStabilizedC3D8R( int elementID )
      : Base( elementID, Marmot::FiniteElement::Quadrature::ReducedIntegration, Base::SectionType::Solid )
    {
      // For URI elements, qps.size() should rigidly equal 1.
      if ( this->qps.size() != 1 ) {
        throw std::invalid_argument(
          "StabilizedC3D8R must be initialized with uniform reduced integration (1 integration point)." );
      }
    }

    void computeYourselfExplicit( const double* QTotal_,
                                  const double* dQ_,
                                  double*       Pe_,
                                  const double* time,
                                  double        dT,
                                  double&       pNewDT )
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
        return;
      }

      S                    = state.stress;
      elasticEnergyDensity = state.elasticEnergyDensity;
      dissipation          = state.dissipation;

      // Subtract the classical under-integrated internal force vector from the RHS
      Pe -= qp.B.transpose() * S * qp.J0xW;

      qp.managedStateVars->elasticStrainEnergy = elasticEnergyDensity * qp.J0xW;
      qp.managedStateVars->dissipation         = dissipation * qp.J0xW;
      qp.managedStateVars->totalStrainEnergy   = ( elasticEnergyDensity + dissipation ) * qp.J0xW;
      qp.managedStateVars->strain += dE;

      // =====================================================================
      // --- GEOMETRY & KINEMATIC SETUP FOR STABILIZATION ---
      // =====================================================================

      // [UPDATED] Map interleaved memory to 3x8 Column-Major Matrices
      // Rows = x, y, z. Columns = Node 0 -> 7.
      using Matrix3x8d = Eigen::Matrix< double, 3, 8 >;
      Map< const Matrix3x8d > U( QTotal.data() );
      Map< Matrix3x8d >       Pe_mat( Pe.data() );

      // Nodal parametric coordinates
      const double xi[8]   = { -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0 };
      const double eta[8]  = { -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0 };
      const double zeta[8] = { -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };

      // Extract central Inverse Jacobian and compute physical volume
      const dNdXiSized    dNdXi0 = this->dNdXi( qp.xi );
      const JacobianSized J0     = this->Jacobian( dNdXi0 );
      const JacobianSized dxi_dx = J0.inverse();

      const double V = 8.0 * qp.detJ; // Physical volume of the element

      // Inline helper array mapping spatial directions to the 6 Voigt stress components
      // ABAQUS ORDER: 0=xx, 1=yy, 2=zz, 3=xy, 4=xz, 5=yz
      const int voigt_map[3][3] = {
        { 0, 3, 4 }, // i=0 (x): 11->0, 12->3, 13->4
        { 3, 1, 5 }, // i=1 (y): 21->3, 22->1, 23->5
        { 4, 5, 2 }  // i=2 (z): 31->4, 32->5, 33->2
      };

      // =====================================================================
      // --- 1. FIRST-ORDER STABILIZATION (f_stab1) ---
      // =====================================================================

      // Pre-compute dense mixed physical-parametric shape function gradients
      double d2N_dXdXi[8][3][3] = { 0.0 };
      for ( int A = 0; A < 8; ++A ) {
        double N21 = 0.125 * xi[A] * eta[A];
        double N31 = 0.125 * xi[A] * zeta[A];
        double N12 = 0.125 * xi[A] * eta[A];
        double N32 = 0.125 * eta[A] * zeta[A];
        double N13 = 0.125 * xi[A] * zeta[A];
        double N23 = 0.125 * eta[A] * zeta[A];

        for ( int i = 0; i < 3; ++i ) {
          d2N_dXdXi[A][i][0] = N21 * dxi_dx( 1, i ) + N31 * dxi_dx( 2, i );
          d2N_dXdXi[A][i][1] = N12 * dxi_dx( 0, i ) + N32 * dxi_dx( 2, i );
          d2N_dXdXi[A][i][2] = N13 * dxi_dx( 0, i ) + N23 * dxi_dx( 1, i );
        }
      }

      using Matrix6x3d       = Eigen::Matrix< double, 6, 3 >;
      Matrix6x3d dStrain_dXi = Matrix6x3d::Zero();

      // [UPDATED] Assemble spatial strain gradients using U(dim, Node)
      for ( int alpha = 0; alpha < 3; ++alpha ) {
        for ( int A = 0; A < 8; ++A ) {
          dStrain_dXi( 0, alpha ) += d2N_dXdXi[A][0][alpha] * U( 0, A );                                      // xx
          dStrain_dXi( 1, alpha ) += d2N_dXdXi[A][1][alpha] * U( 1, A );                                      // yy
          dStrain_dXi( 2, alpha ) += d2N_dXdXi[A][2][alpha] * U( 2, A );                                      // zz
          dStrain_dXi( 3, alpha ) += d2N_dXdXi[A][1][alpha] * U( 0, A ) + d2N_dXdXi[A][0][alpha] * U( 1, A ); // xy
          dStrain_dXi( 4, alpha ) += d2N_dXdXi[A][2][alpha] * U( 0, A ) + d2N_dXdXi[A][0][alpha] * U( 2, A ); // xz
          dStrain_dXi( 5, alpha ) += d2N_dXdXi[A][2][alpha] * U( 1, A ) + d2N_dXdXi[A][1][alpha] * U( 2, A ); // yz
        }
      }

      Matrix6x3d dsigma_dXi = C_alg * dStrain_dXi;

      Matrix3x8d   F_stab1                               = Matrix3x8d::Zero();
      const double secondMomentOfArea_Parametric_Element = V / 3.0;

      for ( int A = 0; A < 8; ++A ) {
        for ( int j = 0; j < 3; ++j ) {
          for ( int i = 0; i < 3; ++i ) {
            int    I           = voigt_map[i][j];
            double contraction = 0.0;
            contraction += d2N_dXdXi[A][i][0] * dsigma_dXi( I, 0 );
            contraction += d2N_dXdXi[A][i][1] * dsigma_dXi( I, 1 );
            contraction += d2N_dXdXi[A][i][2] * dsigma_dXi( I, 2 );
            F_stab1( j, A ) += secondMomentOfArea_Parametric_Element * contraction; // Note index flip: (j, A)
          }
        }
      }

      // =====================================================================
      // --- 2. SECOND-ORDER STABILIZATION (f_stab2) ---
      // =====================================================================

      using Vector8d = Eigen::Matrix< double, 8, 1 >;
      Vector8d Gamma;
      for ( int A = 0; A < 8; ++A ) {
        Gamma( A ) = 0.125 * xi[A] * eta[A] * zeta[A];
      }

      // [UPDATED] U is 3x8, Gamma is 8x1. U * Gamma yields a 3x1 Vector3d directly!
      Eigen::Vector3d u_Gamma = U * Gamma;

      using Vector6d = Eigen::Matrix< double, 6, 1 >;
      Vector6d deps_HG_12, deps_HG_13, deps_HG_23;

      // eps_HG_12 relies on invJ.row(2) (j_3) - ABAQUS ORDER
      deps_HG_12( 0 ) = dxi_dx( 2, 0 ) * u_Gamma( 0 );                                 // xx
      deps_HG_12( 1 ) = dxi_dx( 2, 1 ) * u_Gamma( 1 );                                 // yy
      deps_HG_12( 2 ) = dxi_dx( 2, 2 ) * u_Gamma( 2 );                                 // zz
      deps_HG_12( 3 ) = dxi_dx( 2, 1 ) * u_Gamma( 0 ) + dxi_dx( 2, 0 ) * u_Gamma( 1 ); // xy
      deps_HG_12( 4 ) = dxi_dx( 2, 2 ) * u_Gamma( 0 ) + dxi_dx( 2, 0 ) * u_Gamma( 2 ); // xz
      deps_HG_12( 5 ) = dxi_dx( 2, 2 ) * u_Gamma( 1 ) + dxi_dx( 2, 1 ) * u_Gamma( 2 ); // yz

      // eps_HG_13 relies on invJ.row(1) (j_2) - ABAQUS ORDER
      deps_HG_13( 0 ) = dxi_dx( 1, 0 ) * u_Gamma( 0 );
      deps_HG_13( 1 ) = dxi_dx( 1, 1 ) * u_Gamma( 1 );
      deps_HG_13( 2 ) = dxi_dx( 1, 2 ) * u_Gamma( 2 );
      deps_HG_13( 3 ) = dxi_dx( 1, 1 ) * u_Gamma( 0 ) + dxi_dx( 1, 0 ) * u_Gamma( 1 );
      deps_HG_13( 4 ) = dxi_dx( 1, 2 ) * u_Gamma( 0 ) + dxi_dx( 1, 0 ) * u_Gamma( 2 );
      deps_HG_13( 5 ) = dxi_dx( 1, 2 ) * u_Gamma( 1 ) + dxi_dx( 1, 1 ) * u_Gamma( 2 );

      // eps_HG_23 relies on invJ.row(0) (j_1) - ABAQUS ORDER
      deps_HG_23( 0 ) = dxi_dx( 0, 0 ) * u_Gamma( 0 );
      deps_HG_23( 1 ) = dxi_dx( 0, 1 ) * u_Gamma( 1 );
      deps_HG_23( 2 ) = dxi_dx( 0, 2 ) * u_Gamma( 2 );
      deps_HG_23( 3 ) = dxi_dx( 0, 1 ) * u_Gamma( 0 ) + dxi_dx( 0, 0 ) * u_Gamma( 1 );
      deps_HG_23( 4 ) = dxi_dx( 0, 2 ) * u_Gamma( 0 ) + dxi_dx( 0, 0 ) * u_Gamma( 2 );
      deps_HG_23( 5 ) = dxi_dx( 0, 2 ) * u_Gamma( 1 ) + dxi_dx( 0, 1 ) * u_Gamma( 2 );

      Vector6d dsig_HG_12 = C_alg * deps_HG_12;
      Vector6d dsig_HG_13 = C_alg * deps_HG_13;
      Vector6d dsig_HG_23 = C_alg * deps_HG_23;

      Eigen::Vector3d f_core                = Eigen::Vector3d::Zero();
      const double    fourth_Moment_Of_Area = V / 9.0;

      for ( int j = 0; j < 3; ++j ) {
        for ( int i = 0; i < 3; ++i ) {
          int I = voigt_map[i][j];
          f_core( j ) += dxi_dx( 2, i ) * dsig_HG_12( I ) + dxi_dx( 1, i ) * dsig_HG_13( I ) +
                         dxi_dx( 0, i ) * dsig_HG_23( I );
        }
      }
      f_core *= fourth_Moment_Of_Area;

      Matrix3x8d F_stab2 = f_core * Gamma.transpose();

      // =====================================================================
      // --- 3. FINAL FORCE ASSEMBLY ---
      // =====================================================================

      // Subtract stabilizing forces natively from the mapped 3x8 load vector
      Pe_mat -= ( F_stab1 + F_stab2 );
    }
  };

} // namespace Marmot::Elements
