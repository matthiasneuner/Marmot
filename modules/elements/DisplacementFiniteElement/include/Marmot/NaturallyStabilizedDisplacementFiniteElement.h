/* ---------------------------------------------------------------------
 * Stabilized Hex8 Element (GCD8R - Classical Continuum, Incremental)
 * ---------------------------------------------------------------------
 */
#pragma once

#include "Marmot/DisplacementFiniteElement.h"

namespace Marmot::Elements {

  class NaturallyStabilizedC3D8R : public DisplacementFiniteElement< 3, 8 > {

    double _bulkViscosity{ 0.06 }; // User-defined bulk viscosity parameter for stabilization
    double _charElemLength{ 0.0 }; // Characteristic element length for scaling the bulk viscosity

    // ---------------------------------------------------------------------
    // STABILIZATION HISTORY VARIABLES MAPS
    Eigen::Map< Eigen::Matrix< double, 6, 3 > > _sigma_grad1;
    Eigen::Map< Eigen::Matrix< double, 6, 1 > > _sigma_grad2_12;
    Eigen::Map< Eigen::Matrix< double, 6, 1 > > _sigma_grad2_13;
    Eigen::Map< Eigen::Matrix< double, 6, 1 > > _sigma_grad2_23;

  public:
    using Base = DisplacementFiniteElement< 3, 8 >;

    NaturallyStabilizedC3D8R( int elementID )
      : Base( elementID, Marmot::FiniteElement::Quadrature::ReducedIntegration, Base::SectionType::Solid ),
        _sigma_grad1( nullptr ),
        _sigma_grad2_12( nullptr ),
        _sigma_grad2_13( nullptr ),
        _sigma_grad2_23( nullptr )
    {
      if ( this->qps.size() != 1 ) {
        throw std::invalid_argument(
          "StabilizedC3D8R must be initialized with uniform reduced integration (1 integration point)." );
      }
    }

    void initializeYourself() override
    {
      Base::initializeYourself();

      // Set the characteristic element length for stabilization based on the initial geometry
      const double V  = 8.0 * this->qps[0].detJ; // Physical volume of the element
      _charElemLength = std::cbrt( V );          // Characteristic length as cube root of volume
    }

    void assignStateVars( double* stateVars, int nStateVars ) override
    {
      new ( &_sigma_grad1 ) Eigen::Map< Eigen::Matrix< double, 6, 3 > >( stateVars );
      new ( &_sigma_grad2_12 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 18 );
      new ( &_sigma_grad2_13 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 24 );
      new ( &_sigma_grad2_23 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 30 );

      // Base class state variables (stress, strain, energy, etc.) are stored after the stabilization history variables
      Base::assignStateVars( stateVars + 36, nStateVars - 36 );
    }

    int getNumberOfRequiredStateVars() override
    {
      // 36 state variables for stabilization history + base class state variables
      return 36 + Base::getNumberOfRequiredStateVars();
    }

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

      S = state.stress;

      if ( dT >= 1e-12 ) {
        const double I1dE  = dE.head( 3 ).sum(); // Trace of the strain increment (volumetric part)
        const double p_bv1 = _bulkViscosity * density * dilationalWaveSpeed * _charElemLength * 1. / dT * ( I1dE );
        for ( int i = 0; i < 3; i++ )
          S( i ) += p_bv1;
      }

      // =====================================================================
      // --- DEVIATORIC TANGENT PROJECTION (SYMMETRIC B-BAR) ---
      // =====================================================================
      Eigen::Matrix< double, 6, 6 > P_dev = Eigen::Matrix< double, 6, 6 >::Identity();
      const double                  third = 1.0 / 3.0;
      P_dev( 0, 0 )                       = 2.0 * third;
      P_dev( 0, 1 )                       = -third;
      P_dev( 0, 2 )                       = -third;
      P_dev( 1, 0 )                       = -third;
      P_dev( 1, 1 )                       = 2.0 * third;
      P_dev( 1, 2 )                       = -third;
      P_dev( 2, 0 )                       = -third;
      P_dev( 2, 1 )                       = -third;
      P_dev( 2, 2 )                       = 2.0 * third;

      // Enable to use deviatoric tangent for the stabilization terms only!
      // C_alg = (P_dev * C_alg  ).eval();

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

      using Matrix3x8d = Eigen::Matrix< double, 3, 8 >;
      Map< Matrix3x8d > Pe_mat( Pe.data() );

      // [CRITICAL CHANGE] We now map the INCREMENTAL displacement vector
      Map< const Matrix3x8d > dU( dQ.data() );

      // Nodal parametric coordinates
      const double xi[8]   = { -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0 };
      const double eta[8]  = { -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0 };
      const double zeta[8] = { -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };

      // Extract central Inverse Jacobian and compute physical volume
      const dNdXiSized    dNdXi0 = this->dNdXi( qp.xi );
      const JacobianSized J0     = this->Jacobian( dNdXi0 );
      const JacobianSized dxi_dx = J0.inverse();

      const double V = 8.0 * qp.detJ; // Physical volume of the element

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
      double d2N_dXdXi[8][3][3] = { { { 0.0 } } };
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

      using Matrix6x3d        = Eigen::Matrix< double, 6, 3 >;
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

      // Compute stress gradient INCREMENT and update element history
      Matrix6x3d ddsigma_dXi = C_alg * ddStrain_dXi;
      this->_sigma_grad1 += ddsigma_dXi;

      Matrix3x8d   F_stab1                               = Matrix3x8d::Zero();
      const double secondMomentOfArea_Parametric_Element = V / 3.0;

      // Distribute forces using the TOTAL accumulated stress gradient (_sigma_grad1)
      for ( int A = 0; A < 8; ++A ) {
        for ( int j = 0; j < 3; ++j ) {
          for ( int i = 0; i < 3; ++i ) {
            int    I           = voigt_map[i][j];
            double contraction = 0.0;
            contraction += d2N_dXdXi[A][i][0] * this->_sigma_grad1( I, 0 );
            contraction += d2N_dXdXi[A][i][1] * this->_sigma_grad1( I, 1 );
            contraction += d2N_dXdXi[A][i][2] * this->_sigma_grad1( I, 2 );
            F_stab1( j, A ) += secondMomentOfArea_Parametric_Element * contraction;
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

      // INCREMENTAL hourglass kinematics: dU * Gamma
      Eigen::Vector3d du_Gamma = dU * Gamma;

      using Vector6d = Eigen::Matrix< double, 6, 1 >;
      Vector6d ddeps_HG_12, ddeps_HG_13, ddeps_HG_23;

      // ddeps_HG_12 relies on invJ.row(2) (j_3) - ABAQUS ORDER
      ddeps_HG_12( 0 ) = dxi_dx( 2, 0 ) * du_Gamma( 0 );                                  // xx
      ddeps_HG_12( 1 ) = dxi_dx( 2, 1 ) * du_Gamma( 1 );                                  // yy
      ddeps_HG_12( 2 ) = dxi_dx( 2, 2 ) * du_Gamma( 2 );                                  // zz
      ddeps_HG_12( 3 ) = dxi_dx( 2, 1 ) * du_Gamma( 0 ) + dxi_dx( 2, 0 ) * du_Gamma( 1 ); // xy
      ddeps_HG_12( 4 ) = dxi_dx( 2, 2 ) * du_Gamma( 0 ) + dxi_dx( 2, 0 ) * du_Gamma( 2 ); // xz
      ddeps_HG_12( 5 ) = dxi_dx( 2, 2 ) * du_Gamma( 1 ) + dxi_dx( 2, 1 ) * du_Gamma( 2 ); // yz

      // ddeps_HG_13 relies on invJ.row(1) (j_2) - ABAQUS ORDER
      ddeps_HG_13( 0 ) = dxi_dx( 1, 0 ) * du_Gamma( 0 );
      ddeps_HG_13( 1 ) = dxi_dx( 1, 1 ) * du_Gamma( 1 );
      ddeps_HG_13( 2 ) = dxi_dx( 1, 2 ) * du_Gamma( 2 );
      ddeps_HG_13( 3 ) = dxi_dx( 1, 1 ) * du_Gamma( 0 ) + dxi_dx( 1, 0 ) * du_Gamma( 1 );
      ddeps_HG_13( 4 ) = dxi_dx( 1, 2 ) * du_Gamma( 0 ) + dxi_dx( 1, 0 ) * du_Gamma( 2 );
      ddeps_HG_13( 5 ) = dxi_dx( 1, 2 ) * du_Gamma( 1 ) + dxi_dx( 1, 1 ) * du_Gamma( 2 );

      // ddeps_HG_23 relies on invJ.row(0) (j_1) - ABAQUS ORDER
      ddeps_HG_23( 0 ) = dxi_dx( 0, 0 ) * du_Gamma( 0 );
      ddeps_HG_23( 1 ) = dxi_dx( 0, 1 ) * du_Gamma( 1 );
      ddeps_HG_23( 2 ) = dxi_dx( 0, 2 ) * du_Gamma( 2 );
      ddeps_HG_23( 3 ) = dxi_dx( 0, 1 ) * du_Gamma( 0 ) + dxi_dx( 0, 0 ) * du_Gamma( 1 );
      ddeps_HG_23( 4 ) = dxi_dx( 0, 2 ) * du_Gamma( 0 ) + dxi_dx( 0, 0 ) * du_Gamma( 2 );
      ddeps_HG_23( 5 ) = dxi_dx( 0, 2 ) * du_Gamma( 1 ) + dxi_dx( 0, 1 ) * du_Gamma( 2 );

      // Project increments through tangent and update history
      this->_sigma_grad2_12 += C_alg * ddeps_HG_12;
      this->_sigma_grad2_13 += C_alg * ddeps_HG_13;
      this->_sigma_grad2_23 += C_alg * ddeps_HG_23;

      Eigen::Vector3d f_core                = Eigen::Vector3d::Zero();
      const double    fourth_Moment_Of_Area = V / 9.0;

      // Distribute forces using the TOTAL accumulated hourglass stress gradients
      for ( int j = 0; j < 3; ++j ) {
        for ( int i = 0; i < 3; ++i ) {
          int I = voigt_map[i][j];
          f_core( j ) += dxi_dx( 2, i ) * this->_sigma_grad2_12( I ) + dxi_dx( 1, i ) * this->_sigma_grad2_13( I ) +
                         dxi_dx( 0, i ) * this->_sigma_grad2_23( I );
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
