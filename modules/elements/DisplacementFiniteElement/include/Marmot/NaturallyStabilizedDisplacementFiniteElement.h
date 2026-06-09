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

      // --- GEOMETRY & KINEMATIC SETUP FOR STABILIZATION ---
      // Nodal parametric coordinates
      const double xi[8]   = { -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0 };
      const double eta[8]  = { -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0 };
      const double zeta[8] = { -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };

      // Extract central Inverse Jacobian and compute physical volume
      const dNdXiSized    dNdXi0     = this->dNdXi( qp.xi );
      const JacobianSized J0         = this->Jacobian( dNdXi0 );
      const JacobianSized invJ_eigen = J0.inverse();

      double invJ[3][3];
      for ( int i = 0; i < 3; ++i ) {
        for ( int j = 0; j < 3; ++j ) {
          invJ[i][j] = invJ_eigen( i, j );
        }
      }

      const double V = 8.0 * qp.detJ; // Physical volume of the element

      // Extract total nodal displacements
      double u_A[8][3];
      for ( int A = 0; A < 8; ++A ) {
        u_A[A][0] = QTotal( A * 3 + 0 );
        u_A[A][1] = QTotal( A * 3 + 1 );
        u_A[A][2] = QTotal( A * 3 + 2 );
      }

      // Inline helper array for Voigt mapping
      const int voigt_map[3][3] = {
        { 0, 5, 4 }, // i=0 (x): 11->0, 12->5, 13->4
        { 5, 1, 3 }, // i=1 (y): 21->5, 22->1, 23->3
        { 4, 3, 2 }  // i=2 (z): 31->4, 32->3, 33->2
      };

      // --- 1. FIRST-ORDER STABILIZATION (f_stab1) ---
      // Mixed parametric gradients packed as: [21, 31, 12, 32, 13, 23]
      double N_A_alpha[8][6];
      for ( int A = 0; A < 8; ++A ) {
        N_A_alpha[A][0] = 0.125 * xi[A] * eta[A];
        N_A_alpha[A][1] = 0.125 * xi[A] * zeta[A];
        N_A_alpha[A][2] = 0.125 * xi[A] * eta[A];
        N_A_alpha[A][3] = 0.125 * eta[A] * zeta[A];
        N_A_alpha[A][4] = 0.125 * xi[A] * zeta[A];
        N_A_alpha[A][5] = 0.125 * eta[A] * zeta[A];
      }

      double N_mix[8][3][3] = { 0.0 };
      for ( int A = 0; A < 8; ++A ) {
        for ( int i = 0; i < 3; ++i ) {
          N_mix[A][i][0] = N_A_alpha[A][0] * invJ[1][i] + N_A_alpha[A][1] * invJ[2][i];
          N_mix[A][i][1] = N_A_alpha[A][2] * invJ[0][i] + N_A_alpha[A][3] * invJ[2][i];
          N_mix[A][i][2] = N_A_alpha[A][4] * invJ[0][i] + N_A_alpha[A][5] * invJ[1][i];
        }
      }

      // Spatial gradients of engineering Voigt strain
      double deps_alpha[6][3] = { 0.0 };
      for ( int alpha = 0; alpha < 3; ++alpha ) {
        for ( int A = 0; A < 8; ++A ) {
          deps_alpha[0][alpha] += N_mix[A][0][alpha] * u_A[A][0];
          deps_alpha[1][alpha] += N_mix[A][1][alpha] * u_A[A][1];
          deps_alpha[2][alpha] += N_mix[A][2][alpha] * u_A[A][2];
          deps_alpha[3][alpha] += N_mix[A][2][alpha] * u_A[A][1] + N_mix[A][1][alpha] * u_A[A][2];
          deps_alpha[4][alpha] += N_mix[A][2][alpha] * u_A[A][0] + N_mix[A][0][alpha] * u_A[A][2];
          deps_alpha[5][alpha] += N_mix[A][1][alpha] * u_A[A][0] + N_mix[A][0][alpha] * u_A[A][1];
        }
      }

      // Project strain gradients via implicit algorithmic tangent C_alg
      double dsigma_alpha[6][3] = { 0.0 };
      for ( int alpha = 0; alpha < 3; ++alpha ) {
        for ( int I = 0; I < 6; ++I ) {
          for ( int K = 0; K < 6; ++K ) {
            dsigma_alpha[I][alpha] += C_alg( I, K ) * deps_alpha[K][alpha];
          }
        }
      }

      const double scale1        = V / 3.0;
      double       f_stab1[8][3] = { 0.0 };

      for ( int A = 0; A < 8; ++A ) {
        for ( int j = 0; j < 3; ++j ) {
          for ( int i = 0; i < 3; ++i ) {
            int    I           = voigt_map[i][j];
            double contraction = 0.0;
            contraction += N_mix[A][i][0] * dsigma_alpha[I][0];
            contraction += N_mix[A][i][1] * dsigma_alpha[I][1];
            contraction += N_mix[A][i][2] * dsigma_alpha[I][2];
            f_stab1[A][j] += scale1 * contraction;
          }
        }
      }

      // --- 2. SECOND-ORDER STABILIZATION (f_stab2) ---
      double Gamma[8];
      double u_Gamma[3] = { 0.0, 0.0, 0.0 };
      for ( int A = 0; A < 8; ++A ) {
        Gamma[A] = 0.125 * xi[A] * eta[A] * zeta[A];
        u_Gamma[0] += Gamma[A] * u_A[A][0];
        u_Gamma[1] += Gamma[A] * u_A[A][1];
        u_Gamma[2] += Gamma[A] * u_A[A][2];
      }

      // Fictitious higher-order spatial engineering strain vectors
      double deps_HG_12[6] = { 0 }, deps_HG_13[6] = { 0 }, deps_HG_23[6] = { 0 };

      // eps_HG_12 relies on invJ[2][i] (j_3)
      deps_HG_12[0] = invJ[2][0] * u_Gamma[0];
      deps_HG_12[1] = invJ[2][1] * u_Gamma[1];
      deps_HG_12[2] = invJ[2][2] * u_Gamma[2];
      deps_HG_12[3] = invJ[2][1] * u_Gamma[2] + invJ[2][2] * u_Gamma[1];
      deps_HG_12[4] = invJ[2][0] * u_Gamma[2] + invJ[2][2] * u_Gamma[0];
      deps_HG_12[5] = invJ[2][1] * u_Gamma[0] + invJ[2][0] * u_Gamma[1];

      // eps_HG_13 relies on invJ[1][i] (j_2)
      deps_HG_13[0] = invJ[1][0] * u_Gamma[0];
      deps_HG_13[1] = invJ[1][1] * u_Gamma[1];
      deps_HG_13[2] = invJ[1][2] * u_Gamma[2];
      deps_HG_13[3] = invJ[1][1] * u_Gamma[2] + invJ[1][2] * u_Gamma[1];
      deps_HG_13[4] = invJ[1][0] * u_Gamma[2] + invJ[1][2] * u_Gamma[0];
      deps_HG_13[5] = invJ[1][1] * u_Gamma[0] + invJ[1][0] * u_Gamma[1];

      // eps_HG_23 relies on invJ[0][i] (j_1)
      deps_HG_23[0] = invJ[0][0] * u_Gamma[0];
      deps_HG_23[1] = invJ[0][1] * u_Gamma[1];
      deps_HG_23[2] = invJ[0][2] * u_Gamma[2];
      deps_HG_23[3] = invJ[0][1] * u_Gamma[2] + invJ[0][2] * u_Gamma[1];
      deps_HG_23[4] = invJ[0][0] * u_Gamma[2] + invJ[0][2] * u_Gamma[0];
      deps_HG_23[5] = invJ[0][1] * u_Gamma[0] + invJ[0][0] * u_Gamma[1];

      // Project strains via algorithmic tangent C_alg to obtain stress gradients
      double dsig_HG_12[6] = { 0 }, dsig_HG_13[6] = { 0 }, dsig_HG_23[6] = { 0 };
      for ( int I = 0; I < 6; ++I ) {
        for ( int K = 0; K < 6; ++K ) {
          dsig_HG_12[I] += C_alg( I, K ) * deps_HG_12[K];
          dsig_HG_13[I] += C_alg( I, K ) * deps_HG_13[K];
          dsig_HG_23[I] += C_alg( I, K ) * deps_HG_23[K];
        }
      }

      // Assemble Element Core Correction Vector
      double       f_core[3] = { 0.0, 0.0, 0.0 };
      const double scale2    = V / 9.0;

      for ( int j = 0; j < 3; ++j ) {
        for ( int i = 0; i < 3; ++i ) {
          int I = voigt_map[i][j];
          f_core[j] += invJ[2][i] * dsig_HG_12[I] + invJ[1][i] * dsig_HG_13[I] + invJ[0][i] * dsig_HG_23[I];
        }
        f_core[j] *= scale2;
      }

      // Distribute via pure hourglass filter
      double f_stab2[8][3] = { 0.0 };
      for ( int A = 0; A < 8; ++A ) {
        f_stab2[A][0] += Gamma[A] * f_core[0];
        f_stab2[A][1] += Gamma[A] * f_core[1];
        f_stab2[A][2] += Gamma[A] * f_core[2];
      }

      // --- 3. FINAL FORCE ASSEMBLY ---
      // Subtract the stabilizing internal forces from the RHS load vector Pe
      for ( int A = 0; A < 8; ++A ) {
        Pe( A * 3 + 0 ) -= ( f_stab1[A][0] + f_stab2[A][0] );
        Pe( A * 3 + 1 ) -= ( f_stab1[A][1] + f_stab2[A][1] );
        Pe( A * 3 + 2 ) -= ( f_stab1[A][2] + f_stab2[A][2] );
      }
    }
  };

} // namespace Marmot::Elements
