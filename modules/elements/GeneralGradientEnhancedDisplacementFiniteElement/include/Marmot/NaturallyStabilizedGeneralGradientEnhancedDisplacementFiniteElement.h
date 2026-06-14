/* ---------------------------------------------------------------------
 * Stabilized Gradient-Enhanced Hex8 Element (GC3D8R)
 * ---------------------------------------------------------------------
 */
#pragma once

#include "Marmot/GeneralGradientEnhancedDisplacementFiniteElement.h"

namespace Marmot::Elements {

  template < int nNonlocalVariables = 1 >
  class NaturallyStabilizedGradientEnhancedC3D8R
    : public GeneralGradientEnhancedDisplacementFiniteElement< 3, 8, nNonlocalVariables, 8 > {

    double _bulkViscosity{ 0.06 }; // User-defined bulk viscosity parameter for stabilization
    double _charElemLength{ 0.0 }; // Characteristic element length for scaling the bulk viscosity

    // ---------------------------------------------------------------------
    // STABILIZATION HISTORY VARIABLES MAPS
    Eigen::Map< Eigen::Matrix< double, 6, 3 > > _sigma_grad1;
    Eigen::Map< Eigen::Matrix< double, 6, 1 > > _sigma_grad2_12;
    Eigen::Map< Eigen::Matrix< double, 6, 1 > > _sigma_grad2_13;
    Eigen::Map< Eigen::Matrix< double, 6, 1 > > _sigma_grad2_23;

  public:
    using Base = GeneralGradientEnhancedDisplacementFiniteElement< 3, 8, nNonlocalVariables, 8 >;

    NaturallyStabilizedGradientEnhancedC3D8R( int elementID )
      : Base( elementID, Marmot::FiniteElement::Quadrature::ReducedIntegration, Base::SectionType::Solid ),
        _sigma_grad1( nullptr ),
        _sigma_grad2_12( nullptr ),
        _sigma_grad2_13( nullptr ),
        _sigma_grad2_23( nullptr )
    {
      if ( this->qps.size() != 1 ) {
        throw std::invalid_argument(
          "Stabilized GC3D8R must be initialized with uniform reduced integration (1 integration point)." );
      }
    }

    void initializeYourself() override
    {
      Base::initializeYourself();

      // Set the characteristic element length for stabilization based on the initial geometry
      const double V  = 8.0 * this->qps[0].detJ;
      _charElemLength = std::cbrt( V );
    }

    void assignStateVars( double* stateVars, int nStateVars ) override
    {
      new ( &_sigma_grad1 ) Eigen::Map< Eigen::Matrix< double, 6, 3 > >( stateVars );
      new ( &_sigma_grad2_12 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 18 );
      new ( &_sigma_grad2_13 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 24 );
      new ( &_sigma_grad2_23 ) Eigen::Map< Eigen::Matrix< double, 6, 1 > >( stateVars + 30 );

      // Base class state variables are stored after the 36 stabilization history variables
      Base::assignStateVars( stateVars + 36, nStateVars - 36 );
    }

    int getNumberOfRequiredStateVars() override { return 36 + Base::getNumberOfRequiredStateVars(); }

    void computeYourselfExplicit( const double* QTotal_,
                                  const double* dQ_,
                                  double*       Pe_,
                                  const double* time,
                                  double        dT,
                                  double&       pNewDT )
    {
      using namespace Marmot;
      using namespace ContinuumMechanics::VoigtNotation;

      using response  = typename MarmotMaterialGeneralGradientEnhancedHypoElastic< nNonlocalVariables >::response;
      using tangents  = typename MarmotMaterialGeneralGradientEnhancedHypoElastic< nNonlocalVariables >::tangents;
      using increment = typename MarmotMaterialGeneralGradientEnhancedHypoElastic< nNonlocalVariables >::increment;

      // -----------------------------------------------------------------------
      // --- 0. MEMORY MAPPING AND BASE CLASSICAL EVALUATION ---
      // -----------------------------------------------------------------------

      Map< const typename Base::RhsSized > QTotal( QTotal_ );
      Map< const typename Base::RhsSized > dQ( dQ_ );
      Map< typename Base::RhsSized >       Pe( Pe_ );

      // Map 3x8 Matrices for Mechanical DOFs (Rows = x,y,z | Cols = Node 0->7)
      using Matrix3x8d = Eigen::Matrix< double, 3, 8 >;
      Map< const Matrix3x8d > dU( dQ.data() ); // INCREMENTAL
      Map< Matrix3x8d >       Pe_U_mat( Pe.data() );

      // Map 8xN Matrices for Nonlocal DOFs (Rows = Node 0->7 | Cols = Var 0->N)
      using Matrix8Nd = Eigen::Matrix< double, 8, nNonlocalVariables >;
      Map< const Matrix8Nd > dqK_mat( dQ.data() + Base::sizeDoFU );    // INCREMENTAL
      Map< const Matrix8Nd > qK_mat( QTotal.data() + Base::sizeDoFU ); // TOTAL (Needed for linear PDE stabilization)
      Map< Matrix8Nd >       Pe_K_mat( Pe.data() + Base::sizeDoFU );

      // Extract single quadrature point and fields
      auto&                             qp     = this->qps[0];
      const typename Base::BSized&      B      = qp.B;
      const typename Base::NSizedK&     N_K    = qp.N_K;
      const typename Base::dNdXiSizedK& dNdX_K = qp.dNdX_K;

      typename Base::Voigt dE = B * dQ.head( Base::sizeDoFU );

      Vector< double, nNonlocalVariables > K;
      Vector< double, nNonlocalVariables > dK;
      for ( size_t n = 0; n < nNonlocalVariables; n++ ) {
        K( n )  = N_K * qK_mat.col( n );
        dK( n ) = N_K * dqK_mat.col( n );
      }

      response  res;
      tangents  tan; // Evaluated for algorithmic stabilization
      increment inc;

      try {
        res.stress               = qp.managedStateVars->stress;
        res.elasticEnergyDensity = qp.managedStateVars->elasticStrainEnergy / qp.J0xW;
        res.dissipation          = qp.managedStateVars->dissipation / qp.J0xW;
        res.stateVars            = qp.managedStateVars->materialStateVars.data();
        inc                      = { dE, K, dK, time[1], dT };

        // Material Update
        qp.material->computeStress( res, tan, inc );

        // Add Bulk Viscosity
        if ( dT >= 1e-12 ) {
          const double density             = qp.material->getDensity( res.stateVars );
          const double dilationalWaveSpeed = qp.material->getMaximumWaveSpeed( res );
          const double I1dE                = dE.head( 3 ).sum();
          const double p_bv1 = _bulkViscosity * density * dilationalWaveSpeed * _charElemLength * 1. / dT * ( I1dE );
          for ( int i = 0; i < 3; i++ ) {
            res.stress( i ) += p_bv1;
          }
        }

        // Accumulate baseline internal forces
        Pe.head( Base::sizeDoFU ) -= B.transpose() * res.stress * qp.J0xW;

        for ( int n = 0; n < nNonlocalVariables; n++ ) {
          Pe_K_mat.col( n ) -= ( N_K.transpose() * K( n ) + res.c( n ) * dNdX_K.transpose() * dNdX_K * qK_mat.col( n ) -
                                 N_K.transpose() * res.KLocal( n ) ) *
                               qp.J0xW;
        }
      }
      catch ( Marmot::StressUpdateFailed& e ) {
        pNewDT = 0.25;
        // Material failure -> exit before updating any stabilization histories
        return;
      }

      // Update state
      qp.managedStateVars->stress = res.stress;
      qp.managedStateVars->strain += make3DVoigt< Base::ParentGeometryElement::voigtSize >( dE );
      qp.managedStateVars->elasticStrainEnergy = res.elasticEnergyDensity * qp.J0xW;
      qp.managedStateVars->dissipation         = res.dissipation * qp.J0xW;
      qp.managedStateVars->totalStrainEnergy   = ( res.elasticEnergyDensity + res.dissipation ) * qp.J0xW;

      // -----------------------------------------------------------------------
      // --- GEOMETRY & KINEMATIC SETUP FOR STABILIZATION ---
      // -----------------------------------------------------------------------

      const double xi[8]   = { -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0 };
      const double eta[8]  = { -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0 };
      const double zeta[8] = { -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };

      const typename Base::dNdXiSized    dNdXi0 = this->localGeometryElement.dNdXi( qp.xi );
      const typename Base::JacobianSized J0     = this->localGeometryElement.Jacobian( dNdXi0 );
      const typename Base::JacobianSized invJ   = J0.inverse();

      const double V = 8.0 * qp.detJ;

      // ABAQUS ORDER: 0=xx, 1=yy, 2=zz, 3=xy, 4=xz, 5=yz
      const int voigt_map[3][3] = { { 0, 3, 4 }, { 3, 1, 5 }, { 4, 5, 2 } };

      // -----------------------------------------------------------------------
      // --- 1. FIRST-ORDER STABILIZATION ---
      // -----------------------------------------------------------------------

      double N_mix[8][3][3] = { { { 0.0 } } };
      for ( int A = 0; A < 8; ++A ) {
        double N21 = 0.125 * xi[A] * eta[A], N31 = 0.125 * xi[A] * zeta[A];
        double N12 = 0.125 * xi[A] * eta[A], N32 = 0.125 * eta[A] * zeta[A];
        double N13 = 0.125 * xi[A] * zeta[A], N23 = 0.125 * eta[A] * zeta[A];

        for ( int i = 0; i < 3; ++i ) {
          N_mix[A][i][0] = N21 * invJ( 1, i ) + N31 * invJ( 2, i );
          N_mix[A][i][1] = N12 * invJ( 0, i ) + N32 * invJ( 2, i );
          N_mix[A][i][2] = N13 * invJ( 0, i ) + N23 * invJ( 1, i );
        }
      }

      // [A] INCREMENTAL Mechanical Kinematics
      using Matrix6x3d        = Eigen::Matrix< double, 6, 3 >;
      Matrix6x3d ddStrain_dXi = Matrix6x3d::Zero();
      for ( int alpha = 0; alpha < 3; ++alpha ) {
        for ( int A = 0; A < 8; ++A ) {
          ddStrain_dXi( 0, alpha ) += N_mix[A][0][alpha] * dU( 0, A );
          ddStrain_dXi( 1, alpha ) += N_mix[A][1][alpha] * dU( 1, A );
          ddStrain_dXi( 2, alpha ) += N_mix[A][2][alpha] * dU( 2, A );
          ddStrain_dXi( 3, alpha ) += N_mix[A][1][alpha] * dU( 0, A ) + N_mix[A][0][alpha] * dU( 1, A );
          ddStrain_dXi( 4, alpha ) += N_mix[A][2][alpha] * dU( 0, A ) + N_mix[A][0][alpha] * dU( 2, A );
          ddStrain_dXi( 5, alpha ) += N_mix[A][2][alpha] * dU( 1, A ) + N_mix[A][1][alpha] * dU( 2, A );
        }
      }

      // [B] INCREMENTAL Nonlocal Kinematics
      using MatrixN3   = Eigen::Matrix< double, nNonlocalVariables, 3 >;
      MatrixN3 ddK_dxi = MatrixN3::Zero();
      for ( int A = 0; A < 8; ++A ) {
        double Np[3] = { 0.125 * xi[A], 0.125 * eta[A], 0.125 * zeta[A] };
        for ( int alpha = 0; alpha < 3; ++alpha ) {
          for ( int n = 0; n < nNonlocalVariables; ++n ) {
            ddK_dxi( n, alpha ) += Np[alpha] * dqK_mat( A, n );
          }
        }
      }

      // [C] Project through coupled tangents and accumulate history
      Matrix6x3d ddsigma_alpha = tan.dStressddStrain * ddStrain_dXi + tan.dStressddK * ddK_dxi;
      this->_sigma_grad1 += ddsigma_alpha;

      Matrix3x8d   F_stab1_U = Matrix3x8d::Zero();
      Matrix8Nd    F_stab1_K = Matrix8Nd::Zero();
      const double scale1    = V / 3.0;

      // Distribute Mechanical Force via TOTAL accumulated history
      for ( int A = 0; A < 8; ++A ) {
        for ( int j = 0; j < 3; ++j ) {
          for ( int i = 0; i < 3; ++i ) {
            int I = voigt_map[i][j];
            F_stab1_U( j, A ) += scale1 * N_mix[A][i][0] * this->_sigma_grad1( I, 0 ) +
                                 scale1 * N_mix[A][i][1] * this->_sigma_grad1( I, 1 ) +
                                 scale1 * N_mix[A][i][2] * this->_sigma_grad1( I, 2 );
          }
        }
      }

      // Distribute Nonlocal PDE Force (Remains strictly reliant on total current state)
      for ( int n = 0; n < nNonlocalVariables; ++n ) {
        for ( int alpha = 0; alpha < 3; ++alpha ) {
          for ( int i = 0; i < 3; ++i ) {
            double d2K = 0.0;
            for ( int B = 0; B < 8; ++B )
              d2K += N_mix[B][i][alpha] * qK_mat( B, n );
            double eta_ia = res.c( n ) * d2K;

            for ( int A = 0; A < 8; ++A ) {
              F_stab1_K( A, n ) += scale1 * N_mix[A][i][alpha] * eta_ia;
            }
          }
        }
      }

      // -----------------------------------------------------------------------
      // --- 2. SECOND-ORDER STABILIZATION ---
      // -----------------------------------------------------------------------

      using Vector8d = Eigen::Matrix< double, 8, 1 >;
      Vector8d Gamma;
      for ( int A = 0; A < 8; ++A )
        Gamma( A ) = 0.125 * xi[A] * eta[A] * zeta[A];

      Eigen::Vector3d du_Gamma = dU * Gamma;

      // [A] INCREMENTAL Mechanical Hourglass Strains
      using Vector6d = Eigen::Matrix< double, 6, 1 >;
      Vector6d ddeps_HG_12, ddeps_HG_13, ddeps_HG_23;

      ddeps_HG_12( 0 ) = invJ( 2, 0 ) * du_Gamma( 0 );
      ddeps_HG_12( 1 ) = invJ( 2, 1 ) * du_Gamma( 1 );
      ddeps_HG_12( 2 ) = invJ( 2, 2 ) * du_Gamma( 2 );
      ddeps_HG_12( 3 ) = invJ( 2, 1 ) * du_Gamma( 0 ) + invJ( 2, 0 ) * du_Gamma( 1 );
      ddeps_HG_12( 4 ) = invJ( 2, 2 ) * du_Gamma( 0 ) + invJ( 2, 0 ) * du_Gamma( 2 );
      ddeps_HG_12( 5 ) = invJ( 2, 2 ) * du_Gamma( 1 ) + invJ( 2, 1 ) * du_Gamma( 2 );

      ddeps_HG_13( 0 ) = invJ( 1, 0 ) * du_Gamma( 0 );
      ddeps_HG_13( 1 ) = invJ( 1, 1 ) * du_Gamma( 1 );
      ddeps_HG_13( 2 ) = invJ( 1, 2 ) * du_Gamma( 2 );
      ddeps_HG_13( 3 ) = invJ( 1, 1 ) * du_Gamma( 0 ) + invJ( 1, 0 ) * du_Gamma( 1 );
      ddeps_HG_13( 4 ) = invJ( 1, 2 ) * du_Gamma( 0 ) + invJ( 1, 0 ) * du_Gamma( 2 );
      ddeps_HG_13( 5 ) = invJ( 1, 2 ) * du_Gamma( 1 ) + invJ( 1, 1 ) * du_Gamma( 2 );

      ddeps_HG_23( 0 ) = invJ( 0, 0 ) * du_Gamma( 0 );
      ddeps_HG_23( 1 ) = invJ( 0, 1 ) * du_Gamma( 1 );
      ddeps_HG_23( 2 ) = invJ( 0, 2 ) * du_Gamma( 2 );
      ddeps_HG_23( 3 ) = invJ( 0, 1 ) * du_Gamma( 0 ) + invJ( 0, 0 ) * du_Gamma( 1 );
      ddeps_HG_23( 4 ) = invJ( 0, 2 ) * du_Gamma( 0 ) + invJ( 0, 0 ) * du_Gamma( 2 );
      ddeps_HG_23( 5 ) = invJ( 0, 2 ) * du_Gamma( 1 ) + invJ( 0, 1 ) * du_Gamma( 2 );

      // [B] INCREMENTAL Nonlocal Hourglass Parameters
      using VectorNd   = Eigen::Matrix< double, nNonlocalVariables, 1 >;
      VectorNd dd2K_12 = VectorNd::Zero(), dd2K_13 = VectorNd::Zero(), dd2K_23 = VectorNd::Zero();

      for ( int A = 0; A < 8; ++A ) {
        double n12 = 0.125 * xi[A] * eta[A];
        double n13 = 0.125 * xi[A] * zeta[A];
        double n23 = 0.125 * eta[A] * zeta[A];
        for ( int n = 0; n < nNonlocalVariables; ++n ) {
          dd2K_12( n ) += n12 * dqK_mat( A, n );
          dd2K_13( n ) += n13 * dqK_mat( A, n );
          dd2K_23( n ) += n23 * dqK_mat( A, n );
        }
      }

      // [C] Project through coupled tangents and update history
      this->_sigma_grad2_12 += tan.dStressddStrain * ddeps_HG_12 + tan.dStressddK * dd2K_12;
      this->_sigma_grad2_13 += tan.dStressddStrain * ddeps_HG_13 + tan.dStressddK * dd2K_13;
      this->_sigma_grad2_23 += tan.dStressddStrain * ddeps_HG_23 + tan.dStressddK * dd2K_23;

      // Distribute Mechanical Force via TOTAL accumulated history
      Eigen::Vector3d f_core_U = Eigen::Vector3d::Zero();
      const double    scale2   = V / 9.0;

      for ( int j = 0; j < 3; ++j ) {
        for ( int i = 0; i < 3; ++i ) {
          int I = voigt_map[i][j];
          f_core_U( j ) += invJ( 2, i ) * this->_sigma_grad2_12( I ) + invJ( 1, i ) * this->_sigma_grad2_13( I ) +
                           invJ( 0, i ) * this->_sigma_grad2_23( I );
        }
      }
      f_core_U *= scale2;
      Matrix3x8d F_stab2_U = f_core_U * Gamma.transpose();

      // Distribute Nonlocal PDE Force (Remains strictly reliant on total current state)
      VectorNd k_Gamma     = qK_mat.transpose() * Gamma;
      VectorNd f_core_K    = VectorNd::Zero();
      double   invJ_sq_sum = 0.0;
      for ( int i = 0; i < 3; ++i )
        invJ_sq_sum += invJ( 2, i ) * invJ( 2, i ) + invJ( 1, i ) * invJ( 1, i ) + invJ( 0, i ) * invJ( 0, i );

      for ( int n = 0; n < nNonlocalVariables; ++n ) {
        f_core_K( n ) = scale2 * res.c( n ) * k_Gamma( n ) * invJ_sq_sum;
      }
      Matrix8Nd F_stab2_K = Gamma * f_core_K.transpose();

      // -----------------------------------------------------------------------
      // --- 3. FINAL FORCE ASSEMBLY ---
      // -----------------------------------------------------------------------

      Pe_U_mat -= ( F_stab1_U + F_stab2_U );
      Pe_K_mat -= ( F_stab1_K + F_stab2_K );
    }
  };

} // namespace Marmot::Elements
